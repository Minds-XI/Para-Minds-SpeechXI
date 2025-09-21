from client.entities.dto import VADResponse
from client.ports.message_pub import IMessagePublisher


class AudioSenderUC:
    def __init__(self,
                 message_publisher:IMessagePublisher):
        self.message_publisher = message_publisher
    
    def send(self,frames:VADResponse):
        self.message_publisher.publish(message=frames)
        
