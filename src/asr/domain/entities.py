from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel
from typing import Optional
class TextChunk(BaseModel):
    begin: Optional[float] = None
    end: Optional[float] = None
    sentence: str = ""

@dataclass
class AudioChunkDTO:
    audio_data:np.ndarray

@dataclass
class ASRResponse:
    word:str
    start:float
    end: float

@dataclass
class ASRProcessorResponse:
    sentense:str
    start:float
    end:float