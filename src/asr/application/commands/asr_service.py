
import asyncio
import os

from asr.application.commands.process_stream import StreamProcessor
from asr.application.ports.message import IMessageAudioSubscriber, IMessageTextPublisher
from asr.domain.sentinels import EOS
from asr.application.commands.session import ConnectionManager
from asr.infrastructure.whisper.utils import load_asr_model, load_audio_chunk, online_factory
from loguru import logger
import sys
from asr.domain.entities import AudioChunkDTO, TextChunk
SAMPLING_RATE = 16000

class AsrService:
    def __init__(self,
                args,
                audio_consumer:IMessageAudioSubscriber,
                text_producer:IMessageTextPublisher,
                ):
        self.args = args
        self.asr, self.tokenizer = load_asr_model(args)  # existing init
        self.min_chunk = args.min_chunk_size
        self.warmed = False
        self.sessions = {}
        self._warmup_if_needed()
        self.audio_consumer = audio_consumer
        self.text_producer = text_producer

    def _warmup_if_needed(self):
        msg = ("Whisper is not warmed up. The first chunk "
               "processing may take longer.")
        if self.args.warmup_file:
            if os.path.isfile(self.args.warmup_file):
                a = load_audio_chunk(self.args.warmup_file, 0, 1)
                self.asr.transcribe(a)
                logger.info("Whisper is warmed up.")
                self.warmed = True
            else:
                logger.critical("Warmup file not available. " + msg)
                sys.exit(1)
        else:
            logger.warning(msg)

    def _new_online(self):
        # Fresh state per stream, reusing the shared model + tokenizer
        return online_factory(self.asr, self.args, tokenizer=self.tokenizer, logfile=sys.stderr)
    
    async def _emit_text(self,
                         connection:ConnectionManager):
         while True:
            text_chunk:TextChunk = await connection.read_from_text_queue()
            if text_chunk is EOS:
                break
            if text_chunk:
                logger.info("sending text chunks")
                await self.text_producer.publish(text=text_chunk)


    async def _ensure_session(self, session_id: str) -> ConnectionManager:
        """Create session (conn + processor) once per session_id."""
        if session_id in self.sessions:
            return self.sessions[session_id]["conn"]

        # new session
        asr_processor = self._new_online()
        asr_processor.init()

        conn = ConnectionManager(session_id=session_id, asr_processor=asr_processor)
        stream_processor = StreamProcessor(connection=conn, min_chunk=self.min_chunk)

        # long-lived processor task
        proc_task = asyncio.create_task(stream_processor.process(), name=f"proc-{session_id}")

        # long-lived emitter task
        emitter_task = asyncio.create_task(self._emit_text(conn), name=f"emit-{session_id}")

        self.sessions[session_id] = {
            "conn": conn,
            "proc_task": proc_task,
            "emitter_task": emitter_task
        }

        logger.info(f"[asr] session created: {session_id}")
        return conn

    async def _route_audio(self,audio_event: AudioChunkDTO):
        """Push incoming audio into the session queue (backpressure-aware)."""
        try:
            conn = await self._ensure_session(audio_event.session_id)
            await conn.write_to_audio_queue(audio_event)
            return conn
        except Exception as e:
            logger.exception(f"[asr] enqueue failed for {audio_event.session_id}: {e}")

    async def stream_recognize(self):
        try:
            while True:
                audio_event = await self.audio_consumer.get()
                if audio_event is None:
                    await asyncio.sleep(0.01)
                    continue
                
                _ = await self._route_audio(audio_event)

        except Exception as e:
            logger.exception(f"[asr] {e}")
        finally:
            await self.audio_consumer.close()
            await self.text_producer.flush()
            await self.text_producer.close()
            # Cancel all session tasks
            for s in self.sessions.values():
                for task_name in ["proc_task", "emitter_task"]:
                    task = s.get(task_name)
                    if task and not task.done():
                        task.cancel()