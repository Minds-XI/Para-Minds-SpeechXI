from abc import ABC,abstractmethod
from typing import Optional

from asr.domain.entities import AudioChunkDTO, TextChunk


class IMessageTextPublisher(ABC):
    @abstractmethod
    async def publish(self,text:TextChunk):
        pass
    @abstractmethod
    async def flush(self,timeout:float):
        pass
    
    @abstractmethod
    async def start(self):
        pass
    
    @abstractmethod
    async def close(self):
        pass
    
class IMessageAudioSubscriber(ABC):
    @abstractmethod
    async def get(self)->Optional[AudioChunkDTO]:
        pass
    
    @abstractmethod
    async def start(self):
        pass
    
    @abstractmethod
    async def close(self):
        pass