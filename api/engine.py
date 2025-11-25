from io import StringIO
from typing import Any, Optional

import torch
import whisperx
from fastapi import File, Query, UploadFile

from api.utils import load_audio, write_result
from ui.config import DEFAULT_MODEL
from ui.constants import LANGUAGE_OPTIONS
from ui.model_manager import ModelCache

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def asr(
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True),
    language: Optional[str] = Query(default=None, enum=list(LANGUAGE_OPTIONS.values())),
    initial_prompt: Optional[str] = Query(default=None),
    diarize: bool = Query(default=True),
    return_speaker_embeddings: bool = Query(default=True),
    output: Optional[str] = Query(default="txt", enum=["txt", "srt", "json"]),
):
    # Load whisper model and reset idle timer (cached)
    cache = ModelCache()
    model = cache.load_model(DEFAULT_MODEL, DEFAULT_DEVICE)

    # Config
    options: dict[str, Any] = {}
    if language:
        options["language"] = language
        if language in ["zh", "ja"]:
            options["chunk_size"] = 10
    if initial_prompt:
        options["initial_prompt"] = initial_prompt

    # Load audio file (never cached)
    audio = load_audio(audio_file.file, encode)

    result = model.transcribe(audio, **options)  # type: ignore
    detected_language = result.get("language")

    input_language = str(detected_language or language)
    output_language = language or input_language
    if input_language == output_language:
        align_model, align_metadata = cache.load_align_model(
            input_language, DEFAULT_DEVICE
        )
        print("Aligning segments...")
        result = whisperx.align(
            result["segments"], align_model, align_metadata, audio, DEFAULT_DEVICE
        )
    else:
        print("Skipping alignment, language mismatch...")

    if diarize:
        diarize_model = cache.load_diarize_model(DEFAULT_DEVICE)
        print("Assigning speaker labels...")

        diarize_result = diarize_model(
            audio, return_embeddings=return_speaker_embeddings
        )
        if return_speaker_embeddings:
            diarize_segments, speaker_embeddings = diarize_result
        else:
            diarize_segments, speaker_embeddings = diarize_result, None

        result = whisperx.assign_word_speakers(
            diarize_segments, result, speaker_embeddings
        )

    # Output data into string stream
    output_file = StringIO()
    write_result(result, output_file, output)
    output_file.seek(0)

    return output_file
