from typing import List
from asr.infrastructure.whisper.base import ASRBase
import torch
from loguru import logger
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from asr.infrastructure.whisper.entities import ASRResponse

class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.
    """

    sep = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    def load_model(self, modelsize=None, cache_dir=None, model_dir=None)->WhisperModel:
#        logging.getLogger("faster_whisper").setLevel(logger.level)
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")


        # this worked fast and reliably on NVIDIA L40
        # model = WhisperModel(model_size_or_path, device=self.device, compute_type="float16", download_root=cache_dir)

        # or run on GPU with INT8
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        #model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8
        # tested: works, but slow, appx 10-times than cuda FP16
        model = WhisperModel(modelsize, device="cpu", compute_type="int8") #, download_root="faster-disk-cache-dir/")
        return model

    def transcribe(self, audio, init_prompt="")-> List[Segment]:

        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(audio,
                                                language=self.original_language,
                                                initial_prompt=init_prompt,
                                                beam_size=5,
                                                word_timestamps=True,
                                                condition_on_previous_text=True,
                                                **self.transcribe_kargs)
        return list(segments)
    
    def timestamp_to_words(self,segments:List[Segment])->List[ASRResponse]:
        output = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                token = ASRResponse(start=word.start,
                                    end=word.end,
                                    word=w)
                output.append(token)
        return output

    def segments_end_ts(self, response:List[ASRResponse])->List[float]:
        return [s.end for s in response]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"
