from abc import ABC,abstractmethod
from typing import List

from asr.domain.entities import AudioChunkDTO


class IASRProcessor(ABC):
    @abstractmethod
    def process_audio(self,audio:AudioChunkDTO):
        pass

    @abstractmethod
    def produce_text(self):
        pass
    
    @abstractmethod
    def init(self):
        pass

