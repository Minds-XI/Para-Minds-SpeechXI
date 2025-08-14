import os
import sys
import numpy as np
import pyaudio
import grpc
from dotenv import load_dotenv
from scipy.io import wavfile

# NEW proto imports (match how you generated them for the server)
from asr.grpc_generated.whisper_pb2 import StreamingRequest, SessionConfig, AudioChunk, Control
from asr.grpc_generated import whisper_pb2_grpc

# Your utils
from asr.utils.audio import Frame
from asr.vad.wbtrc import WebRTCVAD

load_dotenv()

SR = 16000           # server expects 16 kHz
CHUNK_MS = 20        # per your code
BYTES_PER_SAMPLE = 2 # int16
CHANNELS = 1

class AudioStreamClient:
    def __init__(self,
                 sample_rate: int = SR,
                 frame_len_ms: int = CHUNK_MS,
                 padding_ms: int = 100):
        self.pa = pyaudio.PyAudio()
        self.sample_rate = sample_rate
        self.frame_len = frame_len_ms
        self.frame_size = int(self.sample_rate * self.frame_len / 1000)  # samples per chunk
        self.counter_id = 0

        # VAD: yields bytes for speech segments/chunks
        self.vad_service = WebRTCVAD(
            sample_rate=self.sample_rate,
            length=self.frame_len,
            padding_duration_ms=padding_ms
        )

    def _open_stream(self):
        return self.pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size,
        )

    def request_generator(self, lang: str, session_id: str):
        """Yield StreamingRequest messages:

        1) SessionConfig (once)
        2) AudioChunk messages (PCM16LE bytes) gated by VAD
        3) Control(EOS) on exit
        """
        # 1) Send config first
        yield StreamingRequest(config=SessionConfig(
            language_code=lang or "en",
            sample_rate_hz=self.sample_rate,
            channels=CHANNELS,
            enable_partials=True,
            session_id=session_id
        ))

        # 2) Stream mic -> VAD -> AudioChunk
        audio_stream = self._open_stream()
        speech_segments = []
        try:
            while True:
                pcm_bytes = audio_stream.read(self.frame_size, exception_on_overflow=False)
                if not isinstance(pcm_bytes, bytes):
                    print("⚠️  Frame is not bytes", file=sys.stderr)
                    continue
                if len(pcm_bytes) != self.frame_size * BYTES_PER_SAMPLE:
                    # Drop malformed frame to keep alignment
                    print(f"⚠️  Skipping bad frame: {len(pcm_bytes)} bytes", file=sys.stderr)
                    continue

                frame_obj = Frame(bytes=pcm_bytes, timestamp=None, duration=None)

                # Your VAD returns zero or more 'result' chunks (bytes) per input frame
                for result in self.vad_service.process_stream(audio=frame_obj):
                    if not result:
                        continue
                    speech_segments.append(result)

                    # Send as AudioChunk to server
                    yield StreamingRequest(
                        audio=AudioChunk(pcm16=result, seq=self.counter_id)
                    )
                    self.counter_id += 1

        except KeyboardInterrupt:
            print("\n[client] mic stopped by user", file=sys.stderr)
        except Exception as e:
            print(f"[client] error: {e}", file=sys.stderr)
        finally:
            try:
                audio_stream.stop_stream()
                audio_stream.close()
            except Exception:
                pass
            try:
                self.pa.terminate()
            except Exception:
                pass

            # 3) Send EOS so server finalizes
            yield StreamingRequest(control=Control(type=Control.EOS))

            # Optional: persist captured speech to a wav (float32 in [-1,1])
            if speech_segments:
                speech_np = np.concatenate([
                    np.frombuffer(seg, dtype=np.int16).astype(np.float32) / 32768.0
                    for seg in speech_segments
                ])
                try:
                    wavfile.write('output_audio.wav', self.sample_rate, speech_np)
                    print("[client] saved output_audio.wav", file=sys.stderr)
                except Exception as e:
                    print(f"[client] could not save wav: {e}", file=sys.stderr)


def main():
    server_ip = os.environ.get('SERVER_IP', '127.0.0.1')
    server_port = os.environ.get('SERVER_PORT', '8080')
    target = f"{server_ip}:{server_port}"

    # Build gRPC stub for the NEW service
    channel = grpc.insecure_channel(target, options=[
        ("grpc.max_receive_message_length", 20 * 1024 * 1024),
    ])
    stub = whisper_pb2_grpc.AsrStub(channel)

    # Create stream generator
    client = AudioStreamClient()
    gen = client.request_generator(lang=os.environ.get('ASR_LANG', 'en'),
                                   session_id=os.environ.get('SESSION_ID', 'mic-1'))

    # Call the bidi RPC
    try:
        for resp in stub.StreamingRecognize(gen, wait_for_ready=True):
            print(("FINAL" if resp.is_final else "PARTIAL") + f": {resp.text}", flush=True)
    except grpc.RpcError as e:
        print(f"[client] RPC failed: {e.code().name} {e.details()}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
