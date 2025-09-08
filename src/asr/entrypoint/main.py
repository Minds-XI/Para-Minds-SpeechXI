import argparse, asyncio
import os
import grpc
from loguru import logger
from asr.core.whisper.utils import add_shared_args
from asr.transport.grpc.generated import whisper_pb2_grpc
from asr.transport.grpc.service import AsrService
from confluent_kafka import Consumer,Producer
from asr.transport.grpc.generated.whisper_pb2 import StreamingResponse,AudioChunk
from confluent_kafka.schema_registry.protobuf import ProtobufSerializer, ProtobufDeserializer
from confluent_kafka.schema_registry import SchemaRegistryClient
from dotenv import load_dotenv
load_dotenv()
EXTERNAL_HOST = os.environ.get("EXTERNAL_HOST")
SCHEMA_REGISTRY_PORT = os.environ.get("SCHEMA_REGISTRY_PORT")
KAFKA_EXTERNAL_PORT = os.environ.get("KAFKA_EXTERNAL_PORT")
SCHEMA_REGISTRY_URL = f"http://{EXTERNAL_HOST}:{SCHEMA_REGISTRY_PORT}"
# Schema Registry client
sr = SchemaRegistryClient({"url": SCHEMA_REGISTRY_URL})

response_serializer = ProtobufSerializer(StreamingResponse, sr)
response_deserializer = ProtobufDeserializer(StreamingResponse)

req_deserializer = ProtobufDeserializer(AudioChunk)

# Create a consumer configuration
raw_audio_consumer_config = {
    'bootstrap.servers': f'{EXTERNAL_HOST}:{KAFKA_EXTERNAL_PORT}',  # Kafka server address
    'group.id': 'asr-group',                                        # Consumer group name
    'auto.offset.reset': 'earliest'                                 # Start reading from beginning
}
raw_audio_consumer = Consumer(raw_audio_consumer_config)
raw_audio_consumer.subscribe(['raw-audio'])
text_producer = Producer({
        "bootstrap.servers": f"{EXTERNAL_HOST}:{KAFKA_EXTERNAL_PORT}",
        # "acks": "0",
        # "enable.idempotence": True,
        "linger.ms": 5,
        })

async def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file")
    add_shared_args(parser)  # from whisper_online
    args = parser.parse_args()
    asr_service = AsrService(args,
               raw_audio_consumer=raw_audio_consumer,
               raw_audio_deserializer=req_deserializer,
               req_topic_name="raw-audio",
               text_producer=text_producer,
               text_producer_topic="text-chunk",
               response_serializer=response_serializer)
   
    await asr_service.StreamingRecognize()

if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
