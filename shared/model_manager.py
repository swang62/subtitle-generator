import gc
from dataclasses import dataclass
from threading import Lock, Thread
from time import sleep, time
from typing import Any, Optional

import torch
import whisperx

from shared.config import HF_TOKEN
from shared.constants import DEFAULT_DEVICE


@dataclass
class Model:
    data: Optional[Any] = None
    align: Optional[Any] = None
    align_metadata: Optional[Any] = None
    diarize: Optional[Any] = None


@dataclass
class Config:
    model_name: Optional[str] = None
    language: Optional[str] = None
    device: Optional[str] = None
    device_align: Optional[str] = None
    device_diarize: Optional[str] = None


class ModelCache:
    """Manages loading and caching of WhisperX models."""

    model_lock = Lock()
    last_activity_time = time()

    def __init__(self):
        self.model = Model()
        self.config = Config()
        pass

    def cache_timeout(self):
        while True:
            sleep(60)
            if time() - ModelCache.last_activity_time > 300:
                with ModelCache.model_lock:
                    self.cleanup()
                    break

    def cleanup(self):
        del self.model, self.config
        if DEFAULT_DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        self.__init__()
        print("Model unloaded due to timeout")

    def load_model(self, model_name: str, device: str):
        ModelCache.last_activity_time = time()
        with ModelCache.model_lock:
            if (
                self.model.data is None
                or model_name != self.config.model_name
                or device != self.config.device
            ):
                print(f"Loading model:{model_name} on device:{device}")
                compute_type = "float16" if device == "cuda" else "int8"

                self.model.data = whisperx.load_model(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    asr_options={"without_timestamps": False},
                )
                self.config.model_name = model_name
                self.config.device = device

                # Only start idle timer when new model is loaded
                Thread(target=self.cache_timeout, daemon=True).start()

        return self.model.data

    def load_align_model(self, language: str, device: str):
        ModelCache.last_activity_time = time()
        with ModelCache.model_lock:
            if (
                self.model.align is None
                or language != self.config.language
                or device != self.config.device_align
            ):
                print(f"Loading alignment model:{language} on device:{device}")
                self.model.align, self.model.align_metadata = whisperx.load_align_model(
                    language_code=language, device=device
                )
                self.config.language = language
                self.config.device_align = device

        return self.model.align, self.model.align_metadata

    def load_diarize_model(self, device: str):
        ModelCache.last_activity_time = time()
        with ModelCache.model_lock:
            if self.model.diarize is None or device != self.config.device_diarize:
                print(f"Loading diarize model on device:{device}")
                try:
                    from whisperx.diarize import DiarizationPipeline

                    self.model.diarize = DiarizationPipeline(
                        model_name="pyannote/speaker-diarization-3.1",
                        use_auth_token=HF_TOKEN,
                        device=device,
                    )
                    self.model.diarize.model.embedding_batch_size = 8
                    self.model.diarize.model.segmentation_batch_size = 8
                    self.config.device_diarize = device
                except Exception as e:
                    print(str(e))
                    raise RuntimeError(
                        "Make sure your HF_TOKEN is correct and you've accepted "
                        "the terms at: https://huggingface.co/pyannote/speaker-diarization-3.1 "
                        "and https://huggingface.co/pyannote/segmentation-3.0"
                    )

        return self.model.diarize
