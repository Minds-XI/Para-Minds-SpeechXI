from client.use_cases import ingest_audio,filter_speech,audio_send
from client.infrastructure.mic_input import PyAudioInputAudio
from client.infrastructure.kafka_pub import KafkaPublisher
from client.infrastructure.vad.wbtrc import WebRTCVAD
from client.entities.dto import ClientConfig

client_config = ClientConfig()
audio_ingester = PyAudioInputAudio(client_config)
vad_service = WebRTCVAD(sample_rate=client_config.sample_rate,
                        length=client_config.frame_len_ms,
                        padding_duration_ms=client_config.padding_ms,
                        mode=client_config.vad_mode)

message_publisher = KafkaPublisher(topic_name=client_config.audio_topic_name,
                           key=client_config.client_id)



ingest_audio_uc = ingest_audio.IngestAudioUC(audio_ingester=audio_ingester)
filter_speech_uc = filter_speech.FilterSpeechUC(vad_operator=vad_service)
audio_sender_uc  = audio_send.AudioSenderUC(message_publisher=message_publisher)