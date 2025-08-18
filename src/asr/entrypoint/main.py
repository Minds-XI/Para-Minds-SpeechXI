import argparse, asyncio
import grpc
from loguru import logger
from asr.core.whisper.utils import add_shared_args
from asr.transport.grpc.generated import whisper_pb2_grpc
from asr.transport.grpc.service import AsrService

async def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
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
