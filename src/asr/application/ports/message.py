from abc import ABC,abstractmethod

from asr.domain.entities import AudioChunkDTO, TextChunk


class IMessageTextPublisher(ABC):
    @abstractmethod
    def publish(self,text:TextChunk):
        pass
    @abstractmethod
    def flush(self,timeout:float):
        pass
    
class IMessageAudioSubscriber(ABC):
    @abstractmethod
    def get(self)->AudioChunkDTO:
        pass
    
    @abstractmethod
    def close(self):
        pass