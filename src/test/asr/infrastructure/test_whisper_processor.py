# test_online_asr_processor.py
import pytest
import numpy as np
from typing import List

from asr.domain.entities import ASRProcessorResponse, AudioChunkDTO
from asr.infrastructure.whisper.base import ASRBase
from asr.infrastructure.whisper.entities import ASRResponse, HypothesisBuffer
from asr.infrastructure.whisper.processor import OnlineASRProcessor

class DummyASR(ASRBase):
    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        return "dummy_model"

    def transcribe(self, audio, init_prompt=""):
        # Returns a list of fake segments for testing
        duration = len(audio)/16000
        words = ["Hello,", "this", "is", "a", "test."]
        ts = [i*0.5 for i in range(len(words))]
        return [(w, ts[i], ts[i]+0.5) for i, w in enumerate(words)]

    def timestamp_to_words(self, res):
        # Convert fake transcribe output to ASRResponse
        return [ASRResponse(word=w[0], start=w[1], end=w[2]) for w in res]

    def use_vad(self):
        return False

    def segments_end_ts(self, response):
        return [r.end for r in response]

@pytest.fixture
def processor():
    dummy_asr = DummyASR(lan="en")
    proc = OnlineASRProcessor(asr=dummy_asr, buffer_trimming=("segment", 2))
    return proc

def fake_audio_chunk():
    # 2 seconds of dummy audio at 16kHz
    return AudioChunkDTO(audio_data=np.zeros(2*16000, dtype=np.float32),
                         seq=0, session_id="test")

def test_process_iter_emits_committed(processor):
    # Feed first chunk
    processor.insert_audio_chunk(fake_audio_chunk().audio_data)
    result1 = processor.process_iter()

    assert result1 is not None, "First process_iter should return committed text"
    assert isinstance(result1, ASRProcessorResponse)

    # Feed second chunk
    processor.insert_audio_chunk(fake_audio_chunk().audio_data)
    result2 = processor.process_iter()

    # This should emit new committed words beyond last_commited_time
    assert result2 is not None
    assert result2.start >= result1.end, "New emitted text should start after previous committed end"

    # Feed empty buffer: should return None
    result3 = processor.process_iter()
    assert result3 is None
