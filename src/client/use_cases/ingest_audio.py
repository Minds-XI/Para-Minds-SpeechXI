from client.ports.audio_input import IAudioInput


class IngestAudioUC:
    def __init__(self,
                 audio_ingester:IAudioInput):
        self.audio_ingester = audio_ingester
    
    def ingest(self):
        frame = self.audio_ingester.listen()
        return frame
    def close(self):
        self.audio_ingester.close()