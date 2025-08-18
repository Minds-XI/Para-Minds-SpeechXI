import os
import sys
import numpy as np
import pyaudio
import grpc
from dotenv import load_dotenv
from scipy.io import wavfile

from asr.transport.grpc.generated.whisper_pb2 import (
    StreamingRequest, SessionConfig, AudioChunk, Control
)
from asr.grpc_generated import whisper_pb2_grpc

from asr.utils.audio import Frame
from asr.vad.wbtrc import WebRTCVAD

load_dotenv()

SR = 16000            # server expects 16 kHz
CHUNK_MS = 20         # 20 ms frames
BYTES_PER_SAMPLE = 2  # int16
CHANNELS = 1

class AudioStreamClient:
    def __init__(self,
                 sample_rate: int = SR,
                 frame_len_ms: int = CHUNK_MS,
                 padding_ms: int = 40,
                 use_vad: bool = False):
        self.pa = pyaudio.PyAudio()
        self.sample_rate = sample_rate
        self.frame_len = frame_len_ms
        self.frame_size = int(self.sample_rate * self.frame_len / 1000)  # samples per chunk
        self.counter_id = 0
        self.use_vad = use_vad

        self.vad_service = WebRTCVAD(
            sample_rate=self.sample_rate,
            length=self.frame_len,
            padding_duration_ms=padding_ms
        ) if use_vad else None

    def _open_stream(self):
        return self.pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size,
        )

    def request_generator(self, lang: str, session_id: str):
        """
        Yields StreamingRequest in this order:
          1) SessionConfig (once)
          2) AudioChunk (PCM16LE bytes) — optionally VAD-gated
          3) Control(EOS) on exit
        """
        # 1) Send config first (server requires first message = config)
        yield StreamingRequest(config=SessionConfig(
            language_code=lang or "en",
            sample_rate_hz=self.sample_rate,
            channels=CHANNELS,
            enable_partials=True,
            session_id=session_id
        ))

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
                    yield StreamingRequest(
                        audio=AudioChunk(pcm16=seg, seq=self.counter_id)
                    )
                    self.counter_id += 1
                    # dumped.append(seg)

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

            # 3) Tell server we're done so it can flush/finalize
            yield StreamingRequest(control=Control(type=Control.EOS))




def main():
    server_ip = os.environ.get('SERVER_IP', '127.0.0.1')
    server_port = os.environ.get('SERVER_PORT', '8080')
    target = f"{server_ip}:{server_port}"

    lang = os.environ.get('ASR_LANG', 'en')
    session_id = os.environ.get('SESSION_ID', 'mic-0')
    # use_vad = os.environ.get('USE_VAD', '0') in ('1', 'true', 'True')

    # Blocking client is fine with an aio server
    with grpc.insecure_channel(target) as channel:
        stub = whisper_pb2_grpc.AsrStub(channel)
        client = AudioStreamClient(use_vad=True)
        gen = client.request_generator(lang=lang, session_id=session_id)

        try:
            # bidi streaming: iterate server responses as they arrive
            for resp in stub.StreamingRecognize(gen, wait_for_ready=True):
                label = "FINAL" if resp.is_final else "PARTIAL"
                # You can also print timestamps: resp.segment_start_ms / segment_end_ms
                print(f"{label}: {resp.text}", flush=True)
        except grpc.RpcError as e:
            print(f"[client] RPC failed: {e.code().name} {e.details()}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
