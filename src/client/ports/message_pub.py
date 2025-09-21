from abc import ABC,abstractmethod

from client.entities.dto import VADResponse


class IMessagePublisher(ABC):
    @abstractmethod
    def publish(self,message:VADResponse):
        pass
