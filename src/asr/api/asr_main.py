from typing import List, Optional
import librosa
import numpy as np
from ray import serve
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from fastapi import  Depends, FastAPI, HTTPException
import io

import soundfile as sf

from asr.domain.entities import ASRModelConfig,ASRRequestDTONaive ,ASRResponseDTO

app = FastAPI()
@serve.deployment(name="asr_model",num_replicas=2,ray_actor_options={"num_cpus":2})
@serve.ingress(app)
class ASRModel:
    def __init__(self,config:ASRModelConfig):
        self.config = config
        self._model = WhisperModel(model_size_or_path=self.config.model_size,
                                    device=self.config.device,
                                    compute_type=self.config.compute_type)
        
    
    async def transcribe(self,
                audio:np.ndarray,
                initial_prompt:Optional[str]=None)->List[ASRResponseDTO]:

        segments,_ = self._model.transcribe(audio=audio,
                                            language=self.config.language,
                                        initial_prompt=initial_prompt,
                                        vad_filter=self.config.vad_filter,
                                        beam_size=self.config.beam_size,
                                        word_timestamps=self.config.word_timestamps,
                                        condition_on_previous_text=self.config.condition_on_previous_text)
        
        # logger.debug(list(segments))
        segments_to_words = self.timestamp_to_words(segments=segments)
        # logger.debug(segments_to_words)
        return segments_to_words
    
    def timestamp_to_words(self,segments:List[Segment])->List[ASRResponseDTO]:
        output = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > self.config.cut_of_no_speech:
                    continue
                w = word.word
                token = ASRResponseDTO(start=word.start,
                                    end=word.end,
                                    word=w)
                output.append(token)
        return output
    
    @app.post("/",response_model=List[ASRResponseDTO])
    async def process(self,asr_dto:ASRRequestDTONaive = Depends())-> List[ASRResponseDTO]:
        content = await  asr_dto.audio.read()
        byte_stream = io.BytesIO(content)
        try:
            # wavfile.read returns (sample_rate, data_array)
            audio,sample_rate = sf.read(byte_stream,dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during conversion")
        
        return await self.transcribe(audio=audio,
                            initial_prompt=asr_dto.initial_prompt)
    

config = ASRModelConfig()

asr_handle = ASRModel.bind(config)
