from abc import ABC,abstractmethod
import numpy as np
class VADServiceBase(ABC):
    @abstractmethod
    def process_stream(self,audio:bytes):
        pass
    
    @abstractmethod
    def process_file(self,file_path:str):
        pass

