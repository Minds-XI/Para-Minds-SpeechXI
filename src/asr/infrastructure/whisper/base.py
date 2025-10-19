from abc import ABC,abstractmethod
import sys
from typing import List, Optional

import numpy as np

from asr.domain.entities import ASRResponseDTO
from asr.infrastructure.whisper.entities import ASRResponse

class ASRBase(ABC):

    sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                # "" for faster-whisper because it emits the spaces when neeeded)
    @abstractmethod
    async def transcribe(self, 
                         audio:np.ndarray,
                        init_prompt:Optional[str]="")-> List[ASRResponseDTO]:
        # raise NotImplemented("must be implemented in the child class")
        pass

    @abstractmethod
    def segments_end_ts(self,response:List[ASRResponse])->List[float]:
        pass
