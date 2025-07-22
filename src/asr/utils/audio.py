from pydub import AudioSegment
import numpy as np
import torch


def read_audio_as_stream(file_path: str, chunk_size: int = 512, sample_rate: int = 16000):
    """
    Read audio file and yield chunks as int16 PCM bytes for VAD processing
    """
    # Read the audio file using pydub
    audio = AudioSegment.from_wav(file_path)
    audio = audio.set_frame_rate(sample_rate)  # Ensure sample rate is correct
    audio = audio.set_channels(1)  # Ensure mono audio for VAD
    
    # Get raw audio samples as int16 (this is what VAD expects)
    audio_samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    
    # print(f"Audio info: {len(audio_samples)} samples, {sample_rate}Hz, {len(audio_samples)/sample_rate:.2f}s")
    
    # Simulate streaming by breaking it into chunks
    num_chunks = len(audio_samples) // chunk_size
    remainder = len(audio_samples) % chunk_size
    
    for i in range(num_chunks):
        chunk = audio_samples[i * chunk_size: (i + 1) * chunk_size]
        chunk_bytes = chunk.tobytes()  # Convert int16 array to bytes
        yield chunk_bytes
    
    # Handle any remaining samples
    if remainder > 0:
        chunk = audio_samples[num_chunks * chunk_size:]
        # Pad with zeros to maintain chunk size if needed
        if len(chunk) < chunk_size:
            padding = np.zeros(chunk_size - len(chunk), dtype=np.int16)
            chunk = np.concatenate([chunk, padding])
        chunk_bytes = chunk.tobytes()
        yield chunk_bytes