import asyncio
from typing import Union
from asr.application.ports.asr_processor import IASRProcessor
from asr.domain.entities import TextChunk
from shared.protos_gen.whisper_pb2 import AudioChunk

class ConnectionManager:
    def __init__(self,
                 session_id:int,
                   asr_processor: IASRProcessor):
        self.session_id = session_id
        self.audio_queue = asyncio.Queue(maxsize=10)
        self.text_queue  = asyncio.Queue(maxsize=10)
        self.asr_processor = asr_processor
        self.last_end = None
        self.is_first = True

    async def write_to_audio_queue(self, item:Union[AudioChunk,object]):  
        await self.audio_queue.put(item)

    async def read_from_audio_queue(self):
        return await self.audio_queue.get()

    async def write_to_text_queue(self, item:Union[TextChunk,object]):  
        await self.text_queue.put(item)

    async def read_from_text_queue(self):
        return await self.text_queue.get()