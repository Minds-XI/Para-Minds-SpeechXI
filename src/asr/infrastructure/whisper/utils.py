import librosa
from functools import lru_cache
import numpy as np
import sys
import time
from loguru import logger

from asr.application.ports.asr_processor import IASRProcessor
from asr.infrastructure.whisper.faster import FasterWhisperASR
from asr.infrastructure.whisper.processor import OnlineASRProcessor, VACOnlineASRProcessor
from asr.infrastructure.whisper.tokenizer import create_tokenizer

def add_shared_args(parser):
    """shared args for simulation (this entry point) and server
    parser: argparse.ArgumentParser object
    """
    parser.add_argument('--min-chunk-size', type=float, default=1.0, help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.')
    parser.add_argument('--model', type=str, default='large-v2', choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(","),help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.")
    parser.add_argument('--model_cache_dir', type=str, default=None, help="Overriding the default model cache dir where models downloaded from the hub are saved")
    parser.add_argument('--model_dir', type=str, default=None, help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
    parser.add_argument('--lan', '--language', type=str, default='auto', help="Source language code, e.g. en,de,cs, or 'auto' for language detection.")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe","translate"],help="Transcribe or translate.")
    parser.add_argument('--backend', type=str, default="faster-whisper", choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"],help='Load only this backend for Whisper processing.')
    parser.add_argument('--vac', action="store_true", default=False, help='Use VAC = voice activity controller. Recommended. Requires torch.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, help='VAC sample size in seconds.')
    parser.add_argument('--vad', action="store_true", default=False, help='Use VAD = voice activity detection, with the default parameters.')
    parser.add_argument('--buffer_trimming', type=str, default="segment", choices=["sentence", "segment"],help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.')
    parser.add_argument('--buffer_trimming_sec', type=float, default=15, help='Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.')
    parser.add_argument("-l", "--log-level", dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the log level", default='DEBUG')


def load_asr_model(args, logfile=sys.stderr):
    """
    Load the heavy ASR backend ONCE (model weights into CPU/GPU)
    and return (asr, tokenizer). No Online* state is created here.
    """
    backend = args.backend
    if backend == "faster-whisper":
        asr_cls = FasterWhisperASR
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    size = args.model
    t0 = time.time()
    logger.info(f"Loading Whisper {size} model for {args.lan}...")
    asr = asr_cls(modelsize=size, lan=args.lan,
                  cache_dir=args.model_cache_dir, model_dir=args.model_dir)
    logger.info(f"Model loaded in {time.time() - t0:.2f}s")

    if getattr(args, "vad", False):
        logger.info("Enabling VAD")
        asr.use_vad()

    # Tokenizer is tiny; OK to create here
    tgt_language = "en" if args.task == "translate" else args.lan
    tokenizer = create_tokenizer(tgt_language) if args.buffer_trimming == "sentence" else None
    return asr, tokenizer


def online_factory(asr, args, tokenizer=None, logfile=sys.stderr)->IASRProcessor:
    """
    Create a NEW OnlineASRProcessor (or VACOnlineASRProcessor) bound to the
    already-loaded `asr`. Use this PER STREAM (per gRPC call).
    """
    if tokenizer is None:
        tgt_language = "en" if args.task == "translate" else args.lan
        tokenizer = create_tokenizer(tgt_language) if args.buffer_trimming == "sentence" else None

    kwargs = dict(logfile=logfile, buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    if getattr(args, "vac", False):
        # VACOnlineASRProcessor expects (min_chunk_size, asr, tokenizer, ...)
        return VACOnlineASRProcessor(args.min_chunk_size, asr, tokenizer, **kwargs)
    else:
        # OnlineASRProcessor expects (asr, tokenizer, ...)
        return OnlineASRProcessor(asr, tokenizer, **kwargs)


@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]
