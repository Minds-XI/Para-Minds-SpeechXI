from abc import ABC,abstractmethod
from client.entities.dto import Frame

class IVADService(ABC):
    @abstractmethod
    def process_stream(self,audio:Frame):
        pass
    
    @abstractmethod
    def process_file(self,file_path:str):
        pass

