import asyncio
import numpy as np
import soundfile as sf
from asr.infrastructure.whisper.faster import FasterWhisperASR

CHUNK_SIZE_SEC = 5  # duration of each chunk to simulate real streaming
AUDIO_PATH = "/mnt/e/Para-Minds-SpeechXI/src/test/asr/target.wav"


async def test_asr_chunked():
    """Test the ASR client by loading an audio file in chunks and sending each chunk."""
    
    # 1️⃣ Load full file once to get sample rate
    with sf.SoundFile(AUDIO_PATH, 'r') as f:
        sr = f.samplerate
        print(f"[INFO] Audio sample rate: {sr}")
        chunk_samples = int(CHUNK_SIZE_SEC * sr)

        # 2️⃣ Initialize ASR client
        asr = FasterWhisperASR()
        asr.sample_rate = sr  # sync client with file SR if needed

        chunk_id = 0
        while True:
            # 3️⃣ Read next chunk
            data = f.read(chunk_samples, dtype='float32')
            if len(data) == 0:
                break

            chunk_id += 1
            print(f"\n[Chunk {chunk_id}] Sending {len(data)} samples...")

            # 4️⃣ Transcribe chunk
            try:
                result = await asr.transcribe(np.array(data), init_prompt="continue")
                if len(result) == 0:
                    print("  ↳ No speech detected.")
                else:
                    print("  ↳ Transcription:", " ".join([w.word for w in result]))
            except Exception as e:
                print(f"  [Error] {e}")

        print("\n✅ All chunks processed successfully!")


if __name__ == "__main__":
    asyncio.run(test_asr_chunked())
