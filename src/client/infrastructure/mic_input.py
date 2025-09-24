import pyaudio
from pyaudio import Stream
from client.entities.dto import Frame,ClientConfig

from client.ports.audio_input import IAudioInput


class PyAudioInputAudio(IAudioInput):
    def __init__(self,config:ClientConfig):
        super().__init__()
        self.pa = pyaudio.PyAudio()
        self.config = config
        self.audio_stream = self._open_stream()
    def _open_stream(self)->Stream:
        return self.pa.open(
            format=pyaudio.paInt16,
            channels=self.config.input_audio_channel,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.frame_size,
        )

    def listen(self)->Frame:

        in_bytes = self.audio_stream.read(self.config.frame_size, exception_on_overflow=False)
        if not isinstance(in_bytes, bytes):
            raise ValueError("Input Audio is not of type bytes")
        if len(in_bytes) != self.config.frame_size * self.config.bytes_per_sample:
            raise ValueError("Bad Frame")
        
        return Frame(bytes=in_bytes)

    def close(self):
        try:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        except Exception:
            pass
        try:
            self.pa.terminate()
        except Exception:
            pass
