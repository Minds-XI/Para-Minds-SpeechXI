from pydantic import BaseModel
from typing import Optional
class TextChunk(BaseModel):
    begin: Optional[float] = None
    end: Optional[float] = None
    sentence: str = ""


