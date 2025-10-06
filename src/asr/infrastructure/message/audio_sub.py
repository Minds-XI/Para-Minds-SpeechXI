import asyncio
from typing import Optional

from loguru import logger
import numpy as np
from asr.application.ports.message import IMessageAudioSubscriber
from asr.domain.entities import AudioChunkDTO, KafkaConfig
from aiokafka import AIOKafkaConsumer
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
            'bootstrap_servers': f'{config.external_host}:{config.schema_registry_port}',  # Kafka server address
            'group_id': 'asr-group',                                        # Consumer group name
            'auto_offset_reset': 'earliest'                                 # Start reading from beginning
        }
        self.raw_audio_consumer = AIOKafkaConsumer(**raw_audio_consumer_config)
        self.topic_name= topic_name
        self.is_started_flag = False

    async def start(self):
        try:
            await self.raw_audio_consumer.start()
            self.raw_audio_consumer.subscribe([self.topic_name])
        except Exception as e:
            logger.error(e)
            return None    
        self.is_started_flag = True
        

    async def get(self)->Optional[AudioChunkDTO]:
        if not self.is_started_flag:
            await self.start()
        try:
            msg = await self.raw_audio_consumer.getone()
        except Exception as e:
            logger.warning(e)
            return None
        
        if msg is None:
            return None
        
        session_id = msg.key.decode() if isinstance(msg.key, bytes) else str(msg.key or "unknown")
        audio_bytes = msg.value() 
        
        audio_event:AudioChunk =  await asyncio.to_thread(
                                    self.deserializer,
                                    audio_bytes,
                                    SerializationContext(topic=self.topic_name, field=MessageField.VALUE)
                                )
        if audio_event is None:
            logger.warning("Failed to deserialize AudioChunk")
            return None
        
        audio_data = np.frombuffer(audio_event.pcm16, dtype="<i2").astype(np.float32) / 32768.0

        return AudioChunkDTO(audio_data=audio_data,
                             seq=audio_event.seq,
                             session_id=session_id)
    
    async def close(self):
        if self.raw_audio_consumer:
            await self.raw_audio_consumer.stop()
            logger.info("Kafka consumer closed.")
