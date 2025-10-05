
import asyncio
import os

from asr.application.pipeline import StreamingPipeline
from asr.application.sentinels import EOS
from asr.application.session import ConnectionManager
from asr.core.whisper.utils import load_asr_model, load_audio_chunk, online_factory
from confluent_kafka import Consumer
from confluent_kafka.schema_registry.protobuf import ProtobufDeserializer
from confluent_kafka.serialization import SerializationContext, MessageField
from confluent_kafka.schema_registry.protobuf import ProtobufSerializer, ProtobufDeserializer
from confluent_kafka import Producer


from loguru import logger
import sys
from asr.domain.entities import TextChunk
from shared.protos_gen.whisper_pb2 import StreamingResponse,AudioChunk
SAMPLING_RATE = 16000

def delivery_report(err, msg):
    if err is not None:
        logger.error(f"[asr] Delivery failed: {err}")
    else:
        logger.info(f"[asr] Delivered message to {msg.topic()} "
                    f"[{msg.partition()}] @ offset {msg.offset()}")


class AsrService:
    def __init__(self,
                args,
                raw_audio_consumer:Consumer,
                raw_audio_deserializer:ProtobufDeserializer,
                req_topic_name:str,
                text_producer:Producer,
                text_producer_topic:str,
                response_serializer:ProtobufSerializer
                 ):
        self.args = args
        self.asr, self.tokenizer = load_asr_model(args)  # existing init
        self.min_chunk = args.min_chunk_size
        self.warmed = False
        self.sessions = {}
        self._warmup_if_needed()
        self.raw_audio_consumer = raw_audio_consumer
        self.raw_audio_deserializer = raw_audio_deserializer
        self.req_topic_name = req_topic_name
        self.text_producer = text_producer
        self.text_producer_topic = text_producer_topic
        self.response_serializer = response_serializer

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
                text_event = StreamingResponse(session_id=connection.session_id,
                                               text=text_chunk.sentence,
                                               is_final=False)
                self.text_producer.produce(
                        self.text_producer_topic,
                        key=connection.session_id,
                        value=self.response_serializer(text_event,
                                                   SerializationContext(self.text_producer_topic, MessageField.VALUE)),
                        callback=delivery_report
                )
                self.text_producer.poll(0)


    async def _ensure_session(self, session_id: str) -> ConnectionManager:
        """Create session (conn + processor) once per session_id."""
        if session_id in self.sessions:
            return self.sessions[session_id]["conn"]

        # new session
        asr_processor = self._new_online()
        asr_processor.init()

        conn = ConnectionManager(session_id=session_id, asr_processor=asr_processor)
        server_processor = StreamingPipeline(connection=conn, min_chunk=self.min_chunk)

        # long-lived processor task
        proc_task = asyncio.create_task(server_processor.process(), name=f"proc-{session_id}")

        # long-lived emitter task
        emitter_task = asyncio.create_task(self._emit_text(conn), name=f"emit-{session_id}")

        self.sessions[session_id] = {
            "conn": conn,
            "proc_task": proc_task,
            "emitter_task": emitter_task
        }

        logger.info(f"[asr] session created: {session_id}")
        return conn

    async def _route_audio(self, session_id: str, audio_evt: AudioChunk):
        """Push incoming audio into the session queue (backpressure-aware)."""
        try:
            conn = await self._ensure_session(session_id)
            await conn.write_to_audio_queue(audio_evt)
            return conn
        except Exception as e:
            logger.exception(f"[asr] enqueue failed for {session_id}: {e}")

    async def StreamingRecognize(self):
        try:
            while True:
                msg = self.raw_audio_consumer.poll(timeout=0.2)
                if msg is None:
                    await asyncio.sleep(0.01)
                    continue
                session_id = msg.key().decode() if msg.key() else "unknown"
                audio_bytes = msg.value() 
                audio_event:AudioChunk = self.raw_audio_deserializer(audio_bytes,SerializationContext(topic=self.req_topic_name,field=MessageField.VALUE))
                # await connection_manager.write_to_audio_queue(audio_event)
                
                _ = await self._route_audio(session_id, audio_event)

        except Exception as e:
            logger.exception(f"[asr] {e}")
        finally:
            self.raw_audio_consumer.close()
            self.text_producer.flush()