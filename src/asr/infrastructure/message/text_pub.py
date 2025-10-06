from typing import Optional
from loguru import logger
from asr.application.ports.message import IMessageTextPublisher
from asr.domain.entities import KafkaConfig, TextChunk
from confluent_kafka.serialization import SerializationContext, MessageField
from shared.protos_gen.whisper_pb2 import StreamingResponse
from confluent_kafka.serialization import SerializationContext, MessageField
from confluent_kafka.schema_registry.protobuf import ProtobufSerializer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka import Producer

def delivery_report(err, msg):
    if err is not None:
        logger.error(f"[asr] Delivery failed: {err}")
    else:
        logger.info(f"[asr] Delivered message to {msg.topic()} "
                    f"[{msg.partition()}] @ offset {msg.offset()}")

class KafkaTextPub(IMessageTextPublisher):
    def __init__(self,
                 config:KafkaConfig,
                 topic_name:str,
                 key:str):
        super().__init__()
        self.producer = Producer({
            "bootstrap.servers": f"{config.external_host}:{config.external_port}",
            "acks": "1",
            "enable.idempotence": True,
            "linger.ms": 5,
        })
        # Schema Registry client
        self.schema_registry_client = SchemaRegistryClient({"url": config.schema_registry_url})
        self.serializer = ProtobufSerializer(StreamingResponse, self.schema_registry_client)
        self.counter_id = 0
        self.topic_name= topic_name
        self.key = key
        
    def publish(self, text:TextChunk)->None:
        text_event = StreamingResponse(session_id=text.session_id,
                                               text=text.sentence,
                                               is_final=False)
        self.producer.produce(
                self.topic_name,
                key=text.session_id,
                value=self.serializer(text_event,
                                            SerializationContext(self.topic_name, MessageField.VALUE)),
                callback=delivery_report
        )
        self.producer.poll(0)

    def flush(self, timeout: Optional[float] = 5.0):
        """
        Flush any buffered messages.
        """
        self.producer.flush(timeout)
        logger.info("Kafka producer flushed.")