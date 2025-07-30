from pydub import AudioSegment
import numpy as np
from typing import Generator

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def read_audio_as_stream(file_path: str, frame_duration_ms: int = 20, sample_rate: int = 16000) -> Generator[Frame, None, None]:
    """
    Read audio file and yield frames as int16 PCM bytes for VAD processing
    Each frame is a duration of 10ms, 20ms, or 30ms.
    """
    try:
        # Frame size calculation in samples
        frame_size = int(sample_rate * (frame_duration_ms / 1000))  # Frame duration in samples
        
        # Read the audio file using pydub
        audio = AudioSegment.from_wav(file_path)
        audio = audio.set_frame_rate(sample_rate)  # Ensure sample rate is correct
        audio = audio.set_channels(1)  # Ensure mono audio for VAD
        
        # Get raw audio samples as int16 (this is what VAD expects)
        audio_samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        
        # Simulate streaming by breaking it into frames
        num_frames = len(audio_samples) // frame_size
        remainder = len(audio_samples) % frame_size
        
        timestamp = 0.0
        duration = float(num_frames) / sample_rate

        # Yield frames
        for i in range(num_frames):
            frame = audio_samples[i * frame_size: (i + 1) * frame_size]
            frame_bytes = frame.tobytes()  # Convert int16 array to bytes
            yield Frame(bytes=frame_bytes,
                        timestamp=timestamp,
                        duration=duration)
                        
            timestamp += duration

        
        # Handle any remaining samples (pad if needed)
        if remainder > 0:
            frame = audio_samples[num_frames * frame_size:]
            if len(frame) < frame_size:
                padding = np.zeros(frame_size - len(frame), dtype=np.int16)
                frame = np.concatenate([frame, padding])
            frame_bytes = frame.tobytes()
            yield Frame(bytes=frame_bytes,
                        timestamp=timestamp,
                        duration=duration)
            
    except Exception as e:
        print(f"Error processing audio file: {e}")
