#!/usr/bin/env python3
import argparse, asyncio, logging, os, sys
import numpy as np
import grpc
from loguru import logger
from asr.core.whisper.utils import add_shared_args, load_asr_model,load_audio_chunk, online_factory
# from whisper_online import *  # your existing module (asr_factory, online, etc.)
from asr.grpc_generated.whisper_pb2 import StreamingResponse
from asr.grpc_generated import whisper_pb2_grpc

SAMPLING_RATE = 16000

def pcm16le_bytes_to_float32(buf: bytes) -> np.ndarray:
    if not buf:
        return np.empty((0,), dtype=np.float32)
    # Ensure even number of bytes (int16 frames)
    if len(buf) & 1:
        buf = buf[:-1]
    a = np.frombuffer(buf, dtype="<i2").astype(np.float32)
    return a / 32768.0


def format_output_transcript(o, last_end_ms):
    # Your original logic, adapted to return fields instead of a string
    if o[0] is None:
        return None, last_end_ms

    beg_ms, end_ms = o[0] * 1000, o[1] * 1000
    if last_end_ms is not None:
        beg_ms = max(beg_ms, last_end_ms)
    last_end_ms = end_ms

    text = o[2]
    # If you have word timestamps inside `o`, map them to WordInfo here
    return {"beg_ms": beg_ms, "end_ms": end_ms, "text": text}, last_end_ms

class AsrService(whisper_pb2_grpc.AsrServicer):
    def __init__(self, args):
        self.args = args
        self.asr, self.tokenizer = load_asr_model(args)  # existing init
        self.min_chunk = args.min_chunk_size
        self.warmed = False
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
    
    async def StreamingRecognize(self, request_iter, context):
        cfg = None
        buffer = np.empty((0,), dtype=np.float32)
        samples_per_min_chunk = int(self.min_chunk * SAMPLING_RATE)
        last_end_ms = None

        online_asr = self._new_online()
        online_asr.init()

        # Optional: serialize decode if backend is not re-entrant
        # self.decode_lock = getattr(self, "decode_lock", asyncio.Lock())

        loop = asyncio.get_running_loop()
        MAX_BUFFER_SAMPLES = SAMPLING_RATE * 10  # ~10s safety cap

        async for req in request_iter:
            if context.done():
                return

            if req.HasField("config"):
                cfg = req.config
                if cfg.sample_rate_hz and cfg.sample_rate_hz != SAMPLING_RATE:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                        f"Only {SAMPLING_RATE} Hz supported")
                if cfg.channels and cfg.channels not in (0, 1, 2):
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                        "Only mono/stereo supported")
                continue

            if cfg is None:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                                    "First message must be SessionConfig")

            if req.HasField("audio"):
                samples = pcm16le_bytes_to_float32(
                    req.audio.pcm16,
                    # channels=(cfg.channels or 1)
                )
                if samples.size == 0:
                    continue

                # Backpressure: coalesce if we’re falling behind
                if buffer.size + samples.size > MAX_BUFFER_SAMPLES:
                    # Drop oldest to keep tail responsive (or just log and keep)
                    drop = (buffer.size + samples.size) - MAX_BUFFER_SAMPLES
                    buffer = buffer[drop:]
                    logger.warning(f"Dropped {drop} samples due to backlog")

                buffer = np.concatenate([buffer, samples])

                while buffer.size >= samples_per_min_chunk:
                    to_process = buffer[:samples_per_min_chunk]
                    buffer = buffer[samples_per_min_chunk:]

                    online_asr.insert_audio_chunk(to_process)

                    # Offload decode to executor (keeps event loop snappy)
                    # If using the lock, wrap the call within:
                    # async with self.decode_lock:
                    o = await loop.run_in_executor(None, online_asr.process_iter)

                    result, last_end_ms = format_output_transcript(o, last_end_ms)
                    if result is not None:
                        yield StreamingResponse(
                            session_id=getattr(cfg, "session_id", ""),
                            is_final=False,
                            text=result["text"],
                            # segment_start_ms=result["beg_ms"],
                            # segment_end_ms=result["end_ms"],
                        )

            if req.HasField("control") and req.control.type == req.control.EOS:
                try:
                    # async with self.decode_lock:
                    o = await loop.run_in_executor(None, online_asr.finish)
                except Exception:
                    o = await loop.run_in_executor(None, online_asr.process_iter)

                result, last_end_ms = format_output_transcript(o, last_end_ms)
                if result is not None:
                    yield StreamingResponse(
                        session_id=getattr(cfg, "session_id", ""),
                        is_final=True,
                        text=result["text"],
                        # segment_start_ms=result["beg_ms"],
                        # segment_end_ms=result["end_ms"],
                    )
                return

        # Client closed without EOS: best-effort flush
        if buffer.size > 0:
            online_asr.insert_audio_chunk(buffer)
        try:
            # async with self.decode_lock:
            o = await loop.run_in_executor(None, online_asr.finish)
        except Exception:
            o = await loop.run_in_executor(None, online_asr.process_iter)

        result, _ = format_output_transcript(o, last_end_ms)
        if result is not None:
            yield StreamingResponse(
                session_id=getattr(cfg, "session_id", "") if cfg else "",
                is_final=True,
                text=result["text"],
                # segment_start_ms=result["beg_ms"],
                # segment_end_ms=result["end_ms"],
            )


async def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file")
    add_shared_args(parser)  # from whisper_online
    args = parser.parse_args()

    server = grpc.aio.server(options=[
        ("grpc.max_send_message_length", 20 * 1024 * 1024),
        ("grpc.max_receive_message_length", 20 * 1024 * 1024),
        ("grpc.keepalive_time_ms", 20000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 10000),
        ("grpc.http2.max_ping_strikes", 0),
    ])
    whisper_pb2_grpc.add_AsrServicer_to_server(AsrService(args), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    logger.info(f"gRPC ASR listening on {args.host}:{args.port}")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
