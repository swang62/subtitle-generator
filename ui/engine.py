import os
from datetime import datetime
from typing import Any

import gradio as gr
import whisperx

from shared.model_manager import ModelCache
from ui.utils import save_to_file

cache = ModelCache()


# Main function to transcribe/translate audio
def generate_subtitles(
    file_name: str,
    output_dir: str,
    language: str | None,
    model_name: str,
    device: str,
    chunk_size: int,
    mode: str,
    progress,
):
    start = datetime.now()
    file_path = os.path.join(output_dir, file_name)

    # Config
    options: dict[str, Any] = {}
    if language:
        options["language"] = language
    options["chunk_size"] = chunk_size

    try:
        # Load whisper model
        model = cache.load_model(model_name, device)
        progress.update(1)  # 1

        # Load audio file (never cached)
        print("Loading in audio...")
        audio = whisperx.load_audio(file_path)
        progress.update(1)  # 2

        # Transcribe or translate
        print("Generating...")
        result = model.transcribe(audio, **options)  # type: ignore
        progress.update(1)  # 3

        # Confirm auto-detection worked
        detected_language = result.get("language")
        if detected_language is None and language is None:
            raise ValueError("Language unable to be detected, please select a language")

        # Make sure alignment is possible
        input_language = str(detected_language or language)
        output_language = language or input_language
        if input_language == output_language:
            align_model, align_metadata = cache.load_align_model(input_language, device)
            print("Aligning segments...")
            result = whisperx.align(
                result["segments"], align_model, align_metadata, audio, device
            )
            del align_model, align_metadata
        else:
            print("Skipping alignment, language mismatch...")
        progress.update(1)  # 4

        # Label speakers for meeting transcritions
        if mode != "generate":
            diarize_model = cache.load_diarize_model(device)
            print("Assigning speaker labels...")
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            del diarize_model, diarize_segments
        progress.update(1)  # 5

        # Output data
        output_format = "srt" if mode == "generate" else "txt"
        output_path = save_to_file(
            result["segments"], file_name, output_dir, output_format
        )
        with open(output_path, "r", encoding="utf-8") as file:
            output_data = file.read()

        # Detect unique speakers
        unique_speakers = [
            segment["speaker"] for segment in result["segments"] if "speaker" in segment
        ]
        unique_speakers_str = ",".join(list(set(unique_speakers)))

        # Time elapsed
        duration = (datetime.now() - start).total_seconds()
        print("Done.")

        return {
            "duration": duration,
            "output_data": output_data,
            "output_path": output_path,
            "unique_speakers": unique_speakers_str,
        }

    except Exception as e:
        raise gr.Error(str(e))
