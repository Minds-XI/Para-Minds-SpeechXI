from dataclasses import dataclass
import os
from fastapi import File, Form, UploadFile
import numpy as np
from pydantic import BaseModel
from typing import Literal, Optional
from dotenv import load_dotenv
load_dotenv()
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
    def __post_init__(self):
        if isinstance(self.start,(np.floating, np.float32, np.float64)):
            self.start = float(self.start)
        if isinstance(self.end,(np.floating, np.float32, np.float64)):
            self.end = float(self.end)
        
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


MODEL_SIZES = Literal[
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "distil-small.en",
    "medium",
    "medium.en",
    "distil-medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-large-v3",
    "large-v3-turbo",
    "turbo"
]


class ASRModelConfig(BaseModel):
    model_size: Optional[MODEL_SIZES] = "tiny.en"
    language: Optional[str] = "en"
    task: Optional[str] = "transcribe"
    vad_filter: Optional[bool] = False
    cut_of_no_speech: Optional[float] = 0.9
    beam_size: Optional[int] =5
    word_timestamps:Optional[bool] =True
    condition_on_previous_text: Optional[bool] =True
    device: Optional[str] = "cpu"
    compute_type: Optional[str]= "int8"


class ASRRequestDTO(BaseModel):
    audio: bytes
    initial_prompt: Optional[str] = ""


class ASRRequestDTONaive:
    def __init__(self,
                audio: UploadFile = File(...),
                initial_prompt: str = Form(default="")):
        self.audio = audio
        self.initial_prompt = initial_prompt


class ASRResponseDTO(ASRResponse):
    pass