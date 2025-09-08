import os
import sys
import numpy as np
import pyaudio
import grpc
from dotenv import load_dotenv
from scipy.io import wavfile

from asr.transport.grpc.generated.whisper_pb2 import (
    AudioChunk
)
from confluent_kafka import Producer
from confluent_kafka.serialization import SerializationContext, MessageField
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.protobuf import ProtobufSerializer, ProtobufDeserializer




from asr.utils.audio import Frame
from asr.vad.wbtrc import WebRTCVAD

load_dotenv()

SR = 16000            # server expects 16 kHz
CHUNK_MS = 20         # 20 ms frames
BYTES_PER_SAMPLE = 2  # int16
CHANNELS = 1
EXTERNAL_HOST = os.environ.get("EXTERNAL_HOST")
SCHEMA_REGISTRY_PORT = os.environ.get("SCHEMA_REGISTRY_PORT")
KAFKA_EXTERNAL_PORT = os.environ.get("KAFKA_EXTERNAL_PORT")
SCHEMA_REGISTRY_URL = f"http://{EXTERNAL_HOST}:{SCHEMA_REGISTRY_PORT}"
# Schema Registry client
sr = SchemaRegistryClient({"url": SCHEMA_REGISTRY_URL})

serializer = ProtobufSerializer(AudioChunk, sr)
deserializer = ProtobufDeserializer(AudioChunk)
class AudioStreamClient:
    def __init__(self,
                 client_id:str,
                 producer:Producer,
                 topic_name:str,
                 sample_rate: int = SR,
                 frame_len_ms: int = CHUNK_MS,
                 padding_ms: int = 40,
                 use_vad: bool = False,
                 ):
        self.client_id = client_id
        self.topic_name =topic_name
        self.producer = producer
        self.pa = pyaudio.PyAudio()
        self.sample_rate = sample_rate
        self.frame_len = frame_len_ms
        self.frame_size = int(self.sample_rate * self.frame_len / 1000)  # samples per chunk
        self.counter_id = 0
        self.use_vad = use_vad

        self.vad_service = WebRTCVAD(
            sample_rate=self.sample_rate,
            length=self.frame_len,
            padding_duration_ms=padding_ms,
            mode=2
        ) if use_vad else None

    def _open_stream(self):
        return self.pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size,
        )

    def request_generator(self):

        audio_stream = self._open_stream()
        # dumped = []  # optional: for saving wav

        try:
            while True:
                # Read 20 ms of int16 PCM
                pcm_bytes = audio_stream.read(self.frame_size, exception_on_overflow=False)
                if not isinstance(pcm_bytes, bytes):
                    print("⚠️  Frame is not bytes", file=sys.stderr)
                    continue
                if len(pcm_bytes) != self.frame_size * BYTES_PER_SAMPLE:
                    print(f"⚠️  Skipping bad frame: {len(pcm_bytes)} bytes", file=sys.stderr)
                    continue

                frames_to_send = [pcm_bytes]
                if self.vad_service is not None:
                    try:
                        vad_out = self.vad_service.process_stream(
                            audio=Frame(bytes=pcm_bytes, timestamp=None, duration=None)
                        )
                        # If VAD returns segments, use them; otherwise send nothing (gated)
                        if vad_out:
                            frames_to_send = [seg for seg in vad_out if seg]
                        else:
                            frames_to_send = []
                    except Exception as e:
                        # Fail-open on VAD errors
                        print(f"[client] VAD error: {e} (sending raw)", file=sys.stderr)
                        frames_to_send = [pcm_bytes]

                for seg in frames_to_send:
                    event_audio=AudioChunk(pcm16=seg, seq=self.counter_id)
                    self.producer.produce(
                        self.topic_name,
                        key=self.client_id,
                        value=serializer(event_audio, SerializationContext(self.topic_name, MessageField.VALUE))
                    )
                    self.counter_id += 1
                    # dumped.append(seg)
                if self.counter_id % 100 == 0:
                    self.producer.flush()

        except KeyboardInterrupt:
            print("\n[client] mic stopped by user", file=sys.stderr)
        except Exception as e:
            print(f"[client] error: {e}", file=sys.stderr)
        finally:
            # Close mic
            try:
                audio_stream.stop_stream()
                audio_stream.close()
            except Exception:
                pass
            try:
                self.pa.terminate()
            except Exception:
                pass


def main():
    # Producer
    producer = Producer({
        "bootstrap.servers": f"{EXTERNAL_HOST}:{KAFKA_EXTERNAL_PORT}",
        "acks": "0",
        # "enable.idempotence": True,
        "linger.ms": 5,
        })
    client = AudioStreamClient(client_id="client_1",
                               producer=producer,
                               topic_name="raw-audio",
                               use_vad= True
                               )
    client.request_generator()

if __name__ == "__main__":
    main()
