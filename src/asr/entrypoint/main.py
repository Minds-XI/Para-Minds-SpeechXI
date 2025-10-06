import argparse, asyncio

from loguru import logger
from asr.infrastructure.whisper.utils import add_shared_args
from asr.application.commands.asr_service import AsrService
from asr.domain.entities import KafkaConfig
from asr.infrastructure.message.audio_sub import KafkaAudioSub
from asr.infrastructure.message.text_pub import KafkaTextPub
from dotenv import load_dotenv
load_dotenv()

async def serve():
    kafka_config = KafkaConfig()
    audio_sub = KafkaAudioSub(config=kafka_config,
                              topic_name="raw_audio")
    text_pub = KafkaTextPub(config=kafka_config,
                            topic_name="text-chunk")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file")
    add_shared_args(parser)  # from whisper_online
    args = parser.parse_args()
    asr_service = AsrService(args,
               audio_consumer=audio_sub,
               text_producer=text_pub,
            )
    await text_pub.start()
    await audio_sub.start()
    logger.info("ASR service started.")

    try:
        await asr_service.stream_recognize()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    finally:
        logger.info("ASR service stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
