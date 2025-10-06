import asyncio
from typing import Optional
from loguru import logger
from asr.application.ports.message import IMessageTextPublisher
from asr.domain.entities import KafkaConfig, TextChunk
from confluent_kafka.serialization import SerializationContext, MessageField
from shared.protos_gen.whisper_pb2 import StreamingResponse
from confluent_kafka.serialization import SerializationContext, MessageField
from confluent_kafka.schema_registry.protobuf import ProtobufSerializer
from confluent_kafka.schema_registry import SchemaRegistryClient
from aiokafka import AIOKafkaProducer

class KafkaTextPub(IMessageTextPublisher):
    def __init__(self,
                 config:KafkaConfig,
                 topic_name:str,
                 key:str):
        super().__init__()
        producer_config = {
            "bootstrap_servers": f"{config.external_host}:{config.external_port}",
            "acks": "1",
            "enable_idempotence": True,
            "linger_ms": 5,
        }
        self.producer = AIOKafkaProducer(**producer_config)
        # Schema Registry client
        self.schema_registry_client = SchemaRegistryClient({"url": config.schema_registry_url})
        self.serializer = ProtobufSerializer(StreamingResponse, self.schema_registry_client)
        self.counter_id = 0
        self.topic_name= topic_name
        self.key = key
        
    async def publish(self, text:TextChunk)->None:
        text_event = StreamingResponse(session_id=text.session_id,
                                               text=text.sentence,
                                               is_final=False)
        value_bytes = await asyncio.to_thread(
            self.serializer, text_event, SerializationContext(self.topic_name, MessageField.VALUE)
        )
        await self.producer.send_and_wait(self.topic_name,value=value_bytes,key=text.session_id)

    async def start(self):
        await self.producer.start()
        logger.info(f"AIOKafkaProducer started for topic {self.topic_name}")

    async def flush(self, timeout: Optional[float] = 5.0):
        """
        Flush any buffered messages.
        """
        await self.producer.flush(timeout)
        logger.info("Kafka producer flushed.")
    
    async def close(self):
        await self.producer.stop()
        logger.info(f"AIOKafkaProducer for {self.topic_name} closed.")