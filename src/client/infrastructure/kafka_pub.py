from client.entities.dto import VADResponse,KafkaConfig
from client.ports.message_pub import IMessagePublisher
from confluent_kafka import Producer
from confluent_kafka.serialization import SerializationContext, MessageField
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.protobuf import ProtobufSerializer
from shared.protos_gen.whisper_pb2 import AudioChunk
class KafkaPublisher(IMessagePublisher):
    def __init__(self,
                 topic_name:str,
                 key:str):
        super().__init__()
        self.producer = Producer({
            "bootstrap.servers": f"{KafkaConfig.external_host}:{KafkaConfig.external_port}",
            "acks": "0",
            # "enable.idempotence": True,
            "linger.ms": 5,
        })
        # Schema Registry client
        self.schema_registry_client = SchemaRegistryClient({"url": KafkaConfig.schema_registry_url})
        self.serializer = ProtobufSerializer(AudioChunk, self.schema_registry_client)
        self.counter_id = 0
        self.topic_name= topic_name
        self.key = key
    def publish(self, message:VADResponse):
        if message.frames is None:
            return

        for frame in message.frames:
            event_audio=AudioChunk(pcm16=frame.bytes,
                                    seq=self.counter_id)
            self.producer.produce(
                self.topic_name,
                key=self.key,
                value=self.serializer(event_audio,
                                       SerializationContext(self.topic_name, MessageField.VALUE))
            )
            self.counter_id +=1
        
