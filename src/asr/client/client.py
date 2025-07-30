import os
import numpy as np
import pvporcupine
import pyaudio
import grpc
from asr.grpc_generated.asr_pb2 import AudioStream
from asr.grpc_generated.asr_pb2_grpc import AsrServiceStub
from dotenv import load_dotenv
from asr.utils.audio import Frame
from asr.vad.wbtrc import WebRTCVAD
from scipy.io import wavfile
load_dotenv()
class AudioStreamClient: 
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.frame_len = 30
        self.sample_rate = 16000
        self.counter_id = 0
        self.vad_service = WebRTCVAD(
                                sample_rate=self.sample_rate,
                                length=self.frame_len,
                                padding_duration_ms=100
                                )
    def generator(self):

        audio_stream = self.pa.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=self.sample_rate,
                                input=True,
                                frames_per_buffer=self.frame_len
                            )
        speech_segments = []
        try:
            while True:
                pcm_bytes = audio_stream.read(self.frame_len, exception_on_overflow=False)
                frame_obj = Frame(bytes=pcm_bytes,timestamp=None,duration=None)
                for result in self.vad_service.process_stream(audio=frame_obj):
                    if result:
                        speech_segments.append(result)
                # yield AudioStream(audio=pcm_bytes,
            #             id=self.counter_id)    
            # self.counter_id += 1
        except KeyboardInterrupt:
            print("Recording stopped. Saving output_audio.wav")
            if speech_segments:
                speechs = [
                    np.frombuffer(speech, dtype=np.int16).astype(np.float32) / 32768.0
                    for speech in speech_segments
                ]
                speechs = np.concatenate(speechs)
                wavfile.write('output_audio.wav', self.sample_rate, speechs)


def main():
    # channel = grpc.insecure_channel('localhost:50051')
    # stub = AsrServiceStub(channel)
    stream = AudioStreamClient()
    # response_iter = stub.processAudio(stream.generator())
    # for response in response_iter:
    #     print(response)
    stream.generator()

if __name__== "__main__":
    main()

