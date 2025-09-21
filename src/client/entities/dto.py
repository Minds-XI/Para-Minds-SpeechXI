from dataclasses import dataclass,field
import os
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Frame:
    """Represents a "frame" of audio data."""
    def __init__(self, bytes,):
        self.bytes = bytes

@dataclass
class ClientConfig:
    sample_rate:int = 16000 # 16HZ
    frame_len_ms:int  = 20  # 20 ms frame len
    input_audio_channel:int  = 1
    bytes_per_sample: int = 2 # 16bit
    padding_ms: int = 40
    vad_mode:int = 2
    audio_topic_name:str = "raw_audio"
    client_id:str = "client_1"
    @property
    def frame_size(self,):
        return int(self.sample_rate * self.frame_len_ms / 1000) 

@dataclass
class VADResponse:
    frames: Optional[List[Frame]] = field(default_factory=list)
    
@dataclass
class KafkaConfig:
    external_host = os.environ.get("EXTERNAL_HOST")
    schema_registry_port = os.environ.get("SCHEMA_REGISTRY_PORT")
    external_port = os.environ.get("KAFKA_EXTERNAL_PORT")

    @property
    def schema_registry_url(self):
        return f"http://{self.external_host}:{self.schema_registry_port}"