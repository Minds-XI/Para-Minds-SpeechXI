from abc import ABC,abstractmethod
from typing import List

from asr.domain.entities import ASRProcessorResponse, AudioChunkDTO


class IASRProcessor(ABC):
    @abstractmethod
    def process_audio(self,audio:AudioChunkDTO):
        pass

    @abstractmethod
    def produce_text(self)->ASRProcessorResponse:
        pass
    
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def finish(self)->ASRProcessorResponse:
        pass
    