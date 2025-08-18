
import asyncio
import os

import grpc
from asr.application.pipeline import StreamingPipeline
from asr.application.sentinels import EOS
from asr.application.session import ConnectionManager
from asr.core.whisper.utils import load_asr_model, load_audio_chunk, online_factory
from asr.transport.grpc.generated import whisper_pb2_grpc
from loguru import logger
import sys
from asr.transport.grpc.generated.whisper_pb2 import StreamingResponse,AudioChunk
SAMPLING_RATE = 16000

class AsrService(whisper_pb2_grpc.AsrServicer):
    def __init__(self, args):
        self.args = args
        self.asr, self.tokenizer = load_asr_model(args)  # existing init
        self.min_chunk = args.min_chunk_size
        self.warmed = False
        self.sessions = {}
        self._warmup_if_needed()

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
    
    async def _audio_produce(self,
                       request_iter,
                       connection:ConnectionManager,
                       ):
        try:
            async for request in request_iter:
                if request.HasField("audio"):
                    await connection.write_to_audio_queue(request.audio)
        except Exception as e:
            logger.exception("Error producing audio chunks: %s", e)
        finally:
            # signal the end of audio to the processor
            await connection.write_to_audio_queue(EOS)

    async def _emit_text(self,
                         connection:ConnectionManager):
         while True:
            text_chunk = await connection.read_from_text_queue()
            if text_chunk is EOS:
                break
            if text_chunk:
                yield StreamingResponse(
                    session_id=connection.session_id,
                    is_final=False,
                    text=text_chunk.sentence,
                    # segment_start_ms=text_chunk.begin,
                    # segment_end_ms=text_chunk.end,
                )


    async def StreamingRecognize(self, request_iter, context):
        cfg = None
        loop = asyncio.get_running_loop()
        connection_manager =None


        # Read the first message (must be config) asynchronously
        try:
            header_req = await request_iter.__anext__()  # <-- async header read
        except StopAsyncIteration:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing config header")
            return

        cfg = header_req.config
        if cfg.sample_rate_hz and cfg.sample_rate_hz != SAMPLING_RATE:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                f"Only {SAMPLING_RATE} Hz supported")
            return 
        if cfg.channels and cfg.channels not in (0, 1, 2):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                "Only mono/stereo supported")
            return
        
        session_id = cfg.session_id 
        if  session_id not in self.sessions:
            asr_processor = self._new_online()
            asr_processor.init()
            connection_manager = ConnectionManager(session_id=session_id,
                                                            asr_processor=asr_processor)
            self.sessions[session_id] = connection_manager
            
        else:
            connection_manager = self.sessions.get(session_id)
            
        server_processor = StreamingPipeline(connection=connection_manager,
                                           min_chunk=self.min_chunk)
        # task 1  to push the received audio to the q
        audio_prod_t = loop.create_task(self._audio_produce(request_iter=request_iter,
                                                   connection=connection_manager),
                                                   name="audio_producer")
        
        # task 2 process the audio and produce text
        audio_proc_t = loop.create_task(server_processor.process(),name="audio_processor")

        # task 3 push the text to the client 
        try:
            # 2) Stream results from the async generator
            async for resp in self._emit_text(connection=connection_manager):
                yield resp

        finally:
            # 3) Ensure cleanup if the client disconnects or the stream ends
            for t in (audio_prod_t, audio_proc_t):
                if not t.done():
                    t.cancel()
            await asyncio.gather(audio_prod_t, audio_proc_t, return_exceptions=True)