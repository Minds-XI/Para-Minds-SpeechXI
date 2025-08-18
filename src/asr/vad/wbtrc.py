import collections
import sys
from typing import Generator
from asr.utils.audio import Frame, read_audio_as_stream
from asr.vad.base import VADServiceBase
import webrtcvad


class WebRTCVAD(VADServiceBase):
    def __init__(self,
                 sample_rate:int,
                 length: int,
                 padding_duration_ms:int,
                 mode:int=1,
                 ):
        super().__init__()
        self.vad = webrtcvad.Vad(mode=mode)
        self.sample_rate = sample_rate
        self.length = length
        self.voiced_frames = []
        self.padding_duration_ms = padding_duration_ms
        self.num_padding_frames = int(self.padding_duration_ms / self.length)
        # We use a deque for our sliding window/ring buffer.
        self.ring_buffer = collections.deque(maxlen=self.num_padding_frames)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        self.triggered = False
    def process_file(self, file_path):
        return super().process_file(file_path)
    
    def process_stream(self, audio:Frame):
         yield from  self._vad_collector(frame=audio,
                                   )


    def _vad_collector(self,
                       frame: Frame,
                       )->Generator[bytes,None,None]:
        """Filters out non-voiced audio frames."""
        is_speech = self.vad.is_speech(frame.bytes, self.sample_rate)
        
        # Debug output for speech detection
        # sys.stdout.write('1' if is_speech else '0')

        if not self.triggered:
            # If not triggered, append frame and check if we should trigger
            self.ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in self.ring_buffer if speech])
            if num_voiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = True
                # sys.stdout.write('+(%s)' % (self.ring_buffer[0][0].timestamp,))
                for f, s in self.ring_buffer:
                    self.voiced_frames.append(f)
                self.ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            self.voiced_frames.append(frame)
            self.ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
            if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                self.triggered = False
                yield b''.join([f.bytes for f in self.voiced_frames])
                self.ring_buffer.clear()
                self.voiced_frames = []

        # sys.stdout.write('\n')

