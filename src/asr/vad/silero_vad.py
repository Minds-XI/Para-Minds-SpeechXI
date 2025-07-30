from collections import deque
from pathlib import Path
import numpy as np
import torch
from asr.vad.base import VADServiceBase
from omegaconf import OmegaConf
from typing import Union
from asr.utils.audio import read_audio_as_stream
from config.paths import CONFIG_DIR
import noisereduce as nr

class SileroVADModel:
    def __init__(self,
                 stream_model: bool = True,
                 sample_rate: int = 16000,
                 chunk_size: int = 512):
        # Load the model and utils from the Silero VAD repository
        self.model_vad, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        # Unpack the tuple returned by torch.hub.load
        (self.get_speech_timestamps, self.save_audio, self.read_audio,
         self.VADIterator, self.collect_chunks) = self.utils
        
        self.model_stream = self.VADIterator(self.model_vad)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
    def process(self, chunks: torch.Tensor, **kwargs):
        """Process audio chunks for streaming VAD"""
        return self.model_stream(chunks, **kwargs)

    def process_file(self,
                    audio: Union[str, torch.Tensor],
                    threshold: float = 0.5,
                    **kwargs) -> torch.Tensor:
        """Process entire audio file for VAD"""
        if isinstance(audio, str):
            wav = self.read_audio(audio, 
                                sampling_rate=self.sample_rate)
        else:
            wav = audio
        
        # Process speech timestamps
        timestamps = self.get_speech_timestamps(wav, self.model_vad, 
                                              threshold=threshold, **kwargs)
        # Collect chunks
        filtered_audio = self.collect_chunks(timestamps, wav)
        return filtered_audio

class SileroVADService(VADServiceBase):
    def __init__(self, processor: SileroVADModel):
        super().__init__()
        # CONFIG_DIR = Path("/mnt/data/projects/Para-Minds-SpeechXI/.conf")
        self.config = OmegaConf.load(CONFIG_DIR / "silero_vad.yaml")
        self._set_attribute_from_dict(self.config)
        
        self.processor = processor
        
        # Calculate frame timing
        self.frame_ms = (self.chunk_size / self.sample_rate) * 1000.0
        self.num_pre_roll_frames = int(self.pre_roll_ms // self.frame_ms)
        
        # Initialize buffers
        self.vad_buffer = np.array([], dtype=np.float32)
        self.audio_buffer = bytearray()  # sent to ASR
        self.ring_buffer = deque(maxlen=self.num_pre_roll_frames)
        
        # State tracking
        self.triggered = False
        self.frames_since_start = 0
        self.max_speech_frames = int(getattr(self, 'max_speech_duration_ms', 10000) // self.frame_ms)  # Max 10 seconds
        self.noise_sample = np.array([])
    def process_stream(self, audio: bytes):
        """Process streaming audio data"""
        if not audio:
            return None
            
        # Convert bytes to numpy array
        pcm_samples = np.frombuffer(audio, dtype=np.int16)
        if pcm_samples.size == 0:
            return None
        
        # Convert to float32 normalized audio
        audio_float32 = pcm_samples.astype(np.float32)  / 32768.0
        # process audio 
        # audio_float32 = self._remove_noise(audio_float32)
        # audio_float32 = self._preprocess_chunks(audio_float32)
        self.vad_buffer = np.concatenate((self.vad_buffer, audio_float32))
        
        results = []
        
        # Process complete chunks
        while len(self.vad_buffer) >= self.chunk_size:
            current_chunk = self.vad_buffer[:self.chunk_size]
            self.vad_buffer = self.vad_buffer[self.chunk_size:]

            # Get VAD prediction from Silero VAD
            speech_segments = self.processor.process(
                torch.tensor(current_chunk, dtype=torch.float32), 
                return_seconds=False
            )
            
            # Convert chunk back to int16 bytes for audio buffer
            chunk_int16 = (current_chunk * 32768.0).astype(np.int16).tobytes()
            
            # Check for speech start/end events
            is_speech_start = (speech_segments is not None and 'start' in speech_segments)
            is_speech_end = (speech_segments is not None and 'end' in speech_segments)
            
            print(f"VAD output: {speech_segments}")  # Debug output
            
            # Handle speech detection logic
            result = self._handle_speech_detection(chunk_int16, is_speech_start, is_speech_end)
            if result:
                results.append(result)
        
        return results if results else None

    def _handle_speech_detection(self, chunk_bytes: bytes, is_speech_start: bool, is_speech_end: bool):
        """Handle the logic for detecting speech start/end"""
        
        if not self.triggered:
            # Always add to ring buffer when not triggered
            self.ring_buffer.append(chunk_bytes)
            
            if is_speech_start:
                # Speech start detected - begin recording
                print("Speech started")
                self.triggered = True
                self.frames_since_start = 0
                
                # Add pre-roll frames from ring buffer to audio buffer
                for buffered_chunk in self.ring_buffer:
                    self.audio_buffer.extend(buffered_chunk)
                self.ring_buffer.clear()
                
                return {"event": "speech_start"}
                
        else:
            # Currently in speech - always add to audio buffer
            self.audio_buffer.extend(chunk_bytes)
            self.frames_since_start += 1
            
            if is_speech_end:
                # Speech end detected by VAD
                print("Speech ended (VAD detected)")
                
                # Prepare the complete audio segment
                complete_audio = bytes(self.audio_buffer)
                
                # Reset state
                self._reset_state()
                
                return {
                    "event": "speech_end",
                    "audio": complete_audio,
                    "duration_ms": len(complete_audio) / 2 / self.sample_rate * 1000,
                    "reason": "vad_detected"
                }
            
            elif self.frames_since_start >= self.max_speech_frames:
                # Force end due to maximum duration
                print("Speech ended (timeout)")
                
                # Prepare the complete audio segment
                complete_audio = bytes(self.audio_buffer)
                
                # Reset state
                self._reset_state()
                
                return {
                    "event": "speech_end",
                    "audio": complete_audio,
                    "duration_ms": len(complete_audio) / 2 / self.sample_rate * 1000,
                    "reason": "timeout"
                }
        
        return None

    def _reset_state(self):
        """Reset the VAD state after processing speech segment"""
        self.triggered = False
        self.frames_since_start = 0
        self.audio_buffer = bytearray()
        self.ring_buffer.clear()
        # Keep VAD buffer for continuity - don't reset it

    def finalize_stream(self):
        """Call this when the audio stream ends to capture any ongoing speech"""
        if self.triggered and len(self.audio_buffer) > 0:
            print("Speech ended (end of stream)")
            
            # Prepare the complete audio segment
            complete_audio = bytes(self.audio_buffer)
            
            # Reset state
            self._reset_state()
            
            return {
                "event": "speech_end",
                "audio": complete_audio,
                "duration_ms": len(complete_audio) / 2 / self.sample_rate * 1000,
                "reason": "end_of_stream"
            }
        return None

    def _set_attribute_from_dict(self, config: dict):
        """Set attributes from configuration dictionary"""
        for key, value in config.items():
            setattr(self, key, value)
    
    def _preprocess_chunks(self,chunk:np.array):
        # print(chunk)
        # remove dc part
        chunk = chunk - np.mean(chunk)
        # normalize
        # max_amp= np.max(chunk)
        # if max_amp > 1e6:
        #     chunk = chunk/max_amp
        # else:
        #     chunk = np.zeros_like(chunk)
        return chunk
    def process_file(self, file_path):
        """Process an entire audio file"""
        return super().process_file(file_path)

    def _remove_noise(self,chunk:np.array):
        clean_chunk = nr.reduce_noise(chunk,sr=self.sample_rate,y_noise=self.noise_sample,prop_decrease=0.5)
        return clean_chunk
# Example usage and testing
if __name__ == "__main__":
    # Initialize the VAD system
    processor = SileroVADModel(stream_model=True)
    vad_service = SileroVADService(processor=processor)
    
    # Process audio stream
    try:
        audio_chunks = read_audio_as_stream(
            file_path="/mnt/data/projects/Para-Minds-SpeechXI/.data/Record (online-voice-recorder.com) (1).wav"
        )
        
        speech_segments = []
        
        for i, chunk in enumerate(audio_chunks):
            results = vad_service.process_stream(audio=chunk)
            
            if results:
                for result in results:
                    if result["event"] == "speech_start":
                        print(f"Chunk {i}: Speech segment started")
                    elif result["event"] == "speech_end":
                        print(f"Chunk {i}: Speech segment ended, "
                              f"duration: {result['duration_ms']:.2f}ms, "
                              f"reason: {result.get('reason', 'unknown')}")
                        speech_segments.append(result["audio"])
        
        # Important: Check for any ongoing speech at the end
        final_result = vad_service.finalize_stream()
        if final_result:
            print(f"Final speech segment: duration: {final_result['duration_ms']:.2f}ms, "
                  f"reason: {final_result['reason']}")
            speech_segments.append(final_result["audio"])
        
        print(f"Total speech segments detected: {len(speech_segments)}")
        # for idx,speech in enumerate(speech_segments):
        #     samples = np.frombuffer(speech, dtype=np.int16).astype(np.float32) / 32768.0
        #     tensor_samples = torch.tensor(samples)
        #     processor.save_audio(f"{idx}.wav",tensor_samples)
    except Exception as e:
        print(f"Error processing audio: {e}")