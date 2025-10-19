import io
from typing import List, Optional
import  soundfile as sf
import numpy as np
from asr.domain.errors import ASRServiceNotRespond
from asr.domain.errors import ASRServiceValueError
from asr.infrastructure.whisper.base import ASRBase
import httpx
from asr.domain.entities import ASRResponseDTO, ASRResponse

class FasterWhisperASR(ASRBase):
    sep = ""
    def __init__(self):
        super().__init__()
        self.url = "http://127.0.0.1:8000/"
        self.sample_rate = 16000

    async def transcribe(self,
                    audio:np.ndarray,
                    init_prompt:Optional[str]="")-> List[ASRResponseDTO]:
        
        # create buffer file
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, self.sample_rate, format="WAV")
        wav_buffer.seek(0)
        # create payload
        async with httpx.AsyncClient(timeout=None) as client:
            files = {"audio":("audio.wav", wav_buffer, "audio/wav")}
            data = {"initial_prompt": init_prompt}
            response = await client.post(self.url, files=files, data=data)
            
        if response.status_code != 200:
            raise ASRServiceNotRespond(
                f"ASR service returned {response.status_code}: {response.text}"
            )
        try:
            parsed_response =  [
                ASRResponseDTO(**entry)
                for entry in response.json()
            ]
        except Exception as e:
            raise ASRServiceValueError(f"Failed to parse ASR response: {e}")
        
        return parsed_response

    def segments_end_ts(self, response:List[ASRResponse])->List[float]:
        return [s.end for s in response]
