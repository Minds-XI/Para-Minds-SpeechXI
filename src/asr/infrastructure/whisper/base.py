from abc import ABC,abstractmethod
import sys
from typing import List

from asr.infrastructure.whisper.entities import ASRResponse

class ASRBase(ABC):

    sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                # "" for faster-whisper because it emits the spaces when neeeded)

    def __init__(self,
                  lan,
                    modelsize=None, 
                    cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    @abstractmethod
    def load_model(self, modelsize, cache_dir):
        # raise NotImplemented("must be implemented in the child class")
        pass
    @abstractmethod
    def transcribe(self, audio, init_prompt=""):
        # raise NotImplemented("must be implemented in the child class")
        pass
    @abstractmethod
    def use_vad(self):
        # raise NotImplemented("must be implemented in the child class")
        pass
    
    @abstractmethod
    def timestamp_to_words(self,segments)->List[ASRResponse]:
        pass

    @abstractmethod
    def segments_end_ts(self,response:List[ASRResponse])->List[float]:
        pass
