from client.entities.dto import Frame,VADResponse
from client.ports.speech_filter import IVADService

class FilterSpeechUC:
    def __init__(self,
                 vad_operator:IVADService):
        self.vad_operator = vad_operator
    
    def filter(self,frame:Frame)->VADResponse:
        speeches = self.vad_operator.process_stream(frame=frame)
        return VADResponse(frames=speeches)
    
