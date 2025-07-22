import argparse
from concurrent import futures
from queue import Queue
import threading
import grpc
from asr.grpc_generated.asr_pb2_grpc import AsrService as AsrServiceG
from asr.grpc_generated.asr_pb2 import TextStream
from asr.grpc_generated import asr_pb2_grpc
import riva.client
from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters

try:
    import riva.client.audio_io
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("Please install pyaudio from https://pypi.org/project/PyAudio")
    exit(1)

def parse_args() -> argparse.Namespace:
    default_device_info = riva.client.audio_io.get_default_input_device_info()
    default_device_index = None if default_device_info is None else default_device_info['index']
    parser = argparse.ArgumentParser(
        description="Streaming transcription from microphone via Riva AI Services",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-device", type=int, default=default_device_index, help="An input audio device to use.")
    parser.add_argument("--list-devices", action="store_true", help="List input audio device indices.")
    parser = add_asr_config_argparse_parameters(parser, profanity_filter=True)
    parser = add_connection_argparse_parameters(parser)
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        help="A number of frames per second in audio streamed from a microphone.",
        default=16000,
    )
    parser.add_argument(
        "--file-streaming-chunk",
        type=int,
        default=1600,
        help="A maximum number of frames in a audio chunk sent to server.",
    )
    args = parser.parse_args()
    return args


def config_asr_service():
    args = parse_args()
    auth = riva.client.Auth(args.ssl_cert, args.use_ssl, args.server, args.metadata)
    asr_service = riva.client.ASRService(auth)
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code=args.language_code,
            model=args.model_name,
            max_alternatives=1,
            profanity_filter=args.profanity_filter,
            enable_automatic_punctuation=args.automatic_punctuation,
            verbatim_transcripts=not args.no_verbatim_transcripts,
            sample_rate_hertz=args.sample_rate_hz,
            audio_channel_count=1,
        ),
        interim_results=True,
    )
    riva.client.add_word_boosting_to_config(config, args.boosted_lm_words, args.boosted_lm_score)
    riva.client.add_endpoint_parameters_to_config(
        config,
        args.start_history,
        args.start_threshold,
        args.stop_history,
        args.stop_history_eou,
        args.stop_threshold,
        args.stop_threshold_eou
    )
    riva.client.add_custom_configuration_to_config(
        config,
        args.custom_configuration
    )
    return asr_service,config,args

class ASRService(AsrServiceG):
    def __init__(self):
        super().__init__()
        self.config  = config_asr_service()
        self.asr_client = self.config[0]
        self.args = self.config[-1]
        self.config= self.config[1]
            # Audio producer thread: fill the queue
    def produce(self,request_iter,audio_queue,end_of_stream):
        try:
            for request in request_iter:
                # print(f"Received chunk: id={request.id}, len={len(request.audio)}")
                if request.audio:
                    audio_queue.put(request.audio)
            audio_queue.put(end_of_stream)
        except Exception as e:
            print(f"Error producing audio chunks: {e}")
            audio_queue.put(end_of_stream)


    def processAudio(self,request_iter,context):
        queue = Queue(maxsize=10)
        end_of_stream = object()

        def audio_generator():
            while True:
                chunk = queue.get()
                if chunk is end_of_stream:
                    break
                yield chunk
        threading.Thread(target=self.produce, daemon=True,args=(request_iter,queue,end_of_stream)).start()
        if request_iter is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty payload")
            return 

        try:
     
            responses = self.asr_client.streaming_response_generator(audio_generator(),streaming_config=self.config)
            riva.client.print_streaming(responses=responses,show_intermediate=True)
            # for response in responses:
            #     for result in response.results:
            #         for alternative in result.alternatives:
            #             yield TextStream(id=1, text=alternative.transcript)
        except Exception as e:
            print(f"Error processing request : {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing request : {str(e)}")
            raise

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    asr_pb2_grpc.add_AsrServiceServicer_to_server(
        ASRService(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("ASR gRPC server running on port 50051")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()