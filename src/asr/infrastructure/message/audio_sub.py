from typing import Optional

from loguru import logger
from asr.application.ports.message import IMessageAudioSubscriber
from asr.domain.entities import AudioChunkDTO, KafkaConfig
from confluent_kafka import Consumer
from confluent_kafka.schema_registry.protobuf import  ProtobufDeserializer
from shared.protos_gen.whisper_pb2 import AudioChunk
from confluent_kafka.serialization import SerializationContext, MessageField

class KafkaAudioSub(IMessageAudioSubscriber):

    def __init__(self,
                config:KafkaConfig,
                topic_name:str,
                ):
        super().__init__()
        self.deserializer = ProtobufDeserializer(AudioChunk)

        # Create a consumer configuration
        raw_audio_consumer_config = {
            'bootstrap.servers': f'{config.external_host}:{config.schema_registry_port}',  # Kafka server address
            'group.id': 'asr-group',                                        # Consumer group name
            'auto.offset.reset': 'earliest'                                 # Start reading from beginning
        }
        self.raw_audio_consumer = Consumer(raw_audio_consumer_config)
        self.raw_audio_consumer.subscribe([topic_name])
        self.topic_name= topic_name
    
    def get(self)->Optional[AudioChunkDTO]:
        msg = self.raw_audio_consumer.poll(timeout=0.2)

        if msg is None:
            return None
        
        if msg.error():
            logger.warning(f"Kafka error: {msg.error()}")
            return None
        
        session_id = msg.key().decode() if msg.key() else "unknown"
        audio_bytes = msg.value() 

        audio_event:AudioChunk = self.deserializer(audio_bytes,
                                                   SerializationContext(topic=self.topic_name,field=MessageField.VALUE))
        if audio_event is None:
            logger.warning("Failed to deserialize AudioChunk")
            return None
        
        return AudioChunkDTO(audio_data=audio_event.pcm16,
                             seq=audio_event.seq,
                             session_id=session_id)
    
    def close(self):
        if self.raw_audio_consumer:
            self.raw_audio_consumer.close()
            logger.info("Kafka consumer closed.")
