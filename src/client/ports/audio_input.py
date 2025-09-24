from abc import ABC,abstractmethod

from client.entities.dto import Frame


class IAudioInput(ABC):
    @abstractmethod
    def listen(self)->Frame:
        pass
    @abstractmethod
    def close(self):
        pass