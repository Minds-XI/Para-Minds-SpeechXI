import os
import pvporcupine
import pyaudio
import grpc
from asr.grpc_generated.asr_pb2 import AudioStream
from asr.grpc_generated.asr_pb2_grpc import AsrServiceStub
from dotenv import load_dotenv
load_dotenv()
class AudioStreamClient: 
    ACCESS_KEY = os.getenv("PORCUPINE_API_KEY")
    MODEL_PATH = "/mnt/d/Minds-Work/asr-service/models/Hey-Minds_en_linux_v3_0_0.ppn"
    def __init__(self):
        self.porcupine = pvporcupine.create(access_key=self.ACCESS_KEY,
                                keyword_paths=[self.MODEL_PATH],
                                sensitivities=[0.5])

        self.frame_len = self.porcupine.frame_length
        self.sample_rate = self.porcupine.sample_rate
        self.pa = pyaudio.PyAudio()
        self.counter_id = 0
    def generator(self):

        audio_stream = self.pa.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=self.sample_rate,
                                input=True,
                                frames_per_buffer=self.frame_len
                            )
        while True:
            pcm_bytes = audio_stream.read(self.frame_len, exception_on_overflow=False)
            yield AudioStream(audio=pcm_bytes,
                        id=self.counter_id)    
            self.counter_id += 1
        
def main():
    channel = grpc.insecure_channel('localhost:50051')
    stub = AsrServiceStub(channel)
    stream = AudioStreamClient()
    response_iter = stub.processAudio(stream.generator())
    for response in response_iter:
        print(response)

if __name__== "__main__":
    main()