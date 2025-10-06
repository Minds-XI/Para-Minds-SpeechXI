from dataclasses import dataclass
import os
import numpy as np
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

class TextChunk(BaseModel):
    begin: Optional[float] = None
    end: Optional[float] = None
    sentence: str = ""
    session_id:str

@dataclass
class AudioChunkDTO:
    audio_data:np.ndarray
    session_id:str
    seq:int
    def __len__(self):
        return len(self.audio_data)

@dataclass
class ASRResponse:
    word:str
    start:float
    end: float

@dataclass
class ASRProcessorResponse:
    sentence:str
    start:float
    end:float


@dataclass
class KafkaConfig:
    external_host:str = os.environ.get("EXTERNAL_HOST")
    schema_registry_port:str = os.environ.get("SCHEMA_REGISTRY_PORT")
    external_port:str = os.environ.get("KAFKA_EXTERNAL_PORT")

    @property
    def schema_registry_url(self):
        return f"http://{self.external_host}:{self.schema_registry_port}"