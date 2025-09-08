import asyncio
from typing import Optional

import numpy as np
from asr.application.sentinels import EOS
from asr.application.session import ConnectionManager
from asr.transport.grpc.generated.whisper_pb2 import AudioChunk
from asr.domain.entities import TextChunk
from loguru import logger
SAMPLING_RATE = 16000
FINAL_CHUNK_TIMEOUT = 0.2  # seconds to wait before forcing flush of last audio

class StreamingPipeline:

    def __init__(   self, 
                    connection:ConnectionManager,
                    min_chunk:int,
                ):
        self.connection = connection  
        self.min_chunk = min_chunk

    async def receive_audio_chunk(self) -> tuple[Optional[np.ndarray], bool]:
        """
        Collects audio until min_chunk is reached or EOS is received.
        Returns (audio_array or EOS, is_eos)
        """
        out = []
        minlimit = self.min_chunk * SAMPLING_RATE
        is_eos = False

        while sum(len(x) for x in out) < minlimit:
            try:
                # Wait for audio chunk with timeout to allow final flush
                item = await asyncio.wait_for(self.connection.read_from_audio_queue(), timeout=FINAL_CHUNK_TIMEOUT)
            except asyncio.TimeoutError:
                # No new audio, flush what we have
                if not out:
                    return None, False
                break

            if item is EOS:
                is_eos = True
                if not out:
                    return EOS, True
                break

            chunk: AudioChunk = item
            if chunk is None:
                continue

            raw = np.frombuffer(chunk.pcm16, dtype="<i2").astype(np.float32) / 32768.0
            out.append(raw)

        if not out:
            return None, is_eos

        conc = np.concatenate(out)
        # Handle first chunk if too small
        if self.connection.is_first and len(conc) < minlimit:
            return None, is_eos

        self.connection.is_first = False
        return conc, is_eos



    def format_output_transcript(self, o: tuple):
        # Expect (beg_s, end_s, text)
        if not o:
            return None
        beg_s, end_s, text = o
        # Drop placeholders / empties
        if beg_s is None or end_s is None:
            return None
        if not text or not text.strip():
            return None

        beg, end = beg_s * 1000, end_s * 1000
        if self.connection.last_end is not None:
            beg = max(beg, self.connection.last_end)
        self.connection.last_end = end

        return TextChunk(begin=beg, end=end, sentence=text)

    async def send_result(self, o):
        text_chunk = self.format_output_transcript(o)
        if text_chunk is not None:
            await self.connection.write_to_text_queue(item=text_chunk)
    
    async def _drain_and_finish(self):
        finalize = getattr(self.connection.asr_processor, "finish", None)
        if callable(finalize):
            outs = finalize()
            for seg in outs or []:
                await self.send_result(seg)

    async def process(self):
        """
        Main streaming loop: handles audio chunks, EOS, and flushes incomplete buffer.
        """
        try:
            while True:
                audio, is_eos = await self.receive_audio_chunk()

                if audio is None:
                    if is_eos:
                        # flush last incomplete buffer
                        await self._drain_and_finish()
                        break
                    continue

                if audio is EOS:
                    # client closed without audio
                    await self._drain_and_finish()
                    continue

                # Insert audio into ASR processor
                self.connection.asr_processor.insert_audio_chunk(audio)

                # Process current buffer (partial + complete segments)
                output = self.connection.asr_processor.process_iter()
                await self.send_result(output)

                if is_eos:
                    # flush remaining audio
                    await self._drain_and_finish()
                    continue

        except BrokenPipeError:
            logger.info("broken pipe -- connection closed?")
        finally:
            # Signal text emitter to finish
            await self.connection.write_to_text_queue(EOS)