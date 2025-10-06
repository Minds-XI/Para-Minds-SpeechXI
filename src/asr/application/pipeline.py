import asyncio
from typing import List, Optional, Tuple

import numpy as np
from asr.domain.sentinels import EOS
from asr.application.commands.session import ConnectionManager
from asr.domain.entities import ASRProcessorResponse, AudioChunkDTO, TextChunk
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

    async def receive_audio_chunk(self) -> Tuple[Optional[AudioChunkDTO], bool]:
        """
        Collects audio until min_chunk is reached or EOS is received.
        Returns (audio_array or EOS, is_eos)
        """
        out:List[AudioChunkDTO] = []
        minlimit = self.min_chunk * SAMPLING_RATE
        is_eos = False

        while sum(len(x) for x in out) < minlimit:
            try:
                # Wait for audio chunk with timeout to allow final flush
                item = await asyncio.wait_for(self.connection.read_from_audio_queue(), timeout=FINAL_CHUNK_TIMEOUT)
            except asyncio.TimeoutError:
                if not out:
                    return None, False
                break

            if item is EOS:
                is_eos = True
                if not out:
                    return EOS, True
                break

            chunk: AudioChunkDTO = item
            if chunk is None:
                continue

            out.append(chunk)

        if not out:
            return None, is_eos

        conc = self.concatenate_audio_chunks(out)
        # Handle first chunk if too small
        if self.connection.is_first and len(conc) < minlimit:
            return None, is_eos

        self.connection.is_first = False
        return conc, is_eos


    def concatenate_audio_chunks(self,chunks:List[AudioChunkDTO])->AudioChunkDTO:
        if chunks is None:
            return None
        
        seq = chunks[-1].seq
        session_id = chunks[0].session_id

        res_np = np.concatenate([chunk.audio_data for chunk in chunks])

    
        return AudioChunkDTO(audio_data=res_np,
                             session_id=session_id,
                             seq=seq)
    
    def format_output_transcript(self, response: ASRProcessorResponse):
        # Expect (beg_s, end_s, text)
        if not response:
            return None
        # beg_s, end_s, text = o
        # Drop placeholders / empties
        if response.start is None or response.end is None:
            return None
        if not response.sentence or not response.sentence.strip():
            return None

        beg, end = response.start * 1000, response.end * 1000
        if self.connection.last_end is not None:
            beg = max(beg, self.connection.last_end)
        self.connection.last_end = end

        return TextChunk(begin=beg, end=end, sentence=response.sentence,session_id=self.connection.session_id)

    async def send_result(self, response: ASRProcessorResponse):
        text_chunk = self.format_output_transcript(response)
        if text_chunk is not None:
            await self.connection.write_to_text_queue(item=text_chunk)
    
    async def _drain_and_finish(self):
        response = self.connection.asr_processor.finish()
        await self.send_result(response=response)

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
                self.connection.asr_processor.process_audio(audio)

                # Process current buffer (partial + complete segments)
                output = self.connection.asr_processor.produce_text()
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