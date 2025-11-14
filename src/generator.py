import os
import whisperx
import gradio as gr
from src.model_manager import cache
from src.utils import format_to_minutes, save_to_srt, save_to_txt
from datetime import datetime


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
    options = {}
    if language:
        options["language"] = language
    options["chunk_size"] = chunk_size

    try:
        # Load whisper model
        model = cache.load_model(model_name, device)
        progress.update(1)  # 1

        # Load audio file
        print("Loading in audio...")
        audio = cache.load_audio(file_path)
        progress.update(1)  # 2

        # Transcribe or translate
        print("Generating...")
        output = model.transcribe(audio, **options)  # type: ignore
        progress.update(1)  # 3

        # Confirm auto-detection worked
        detected_language = output.get("language")
        if detected_language is None and language is None:
            raise ValueError("Language unable to be detected, please select a language")

        # Make sure alignment is possible
        input_language = str(detected_language or language)
        output_language = language or input_language
        if input_language == output_language:
            print("Aligning segments...")

            align_model, align_metadata = cache.load_align_model(input_language, device)
            output = whisperx.align(
                output["segments"], align_model, align_metadata, audio, device
            )
        else:
            print("Skipping alignment, language mismatch...")
        progress.update(1)  # 4

        if mode == "generate":
            output_path = save_to_srt(output["segments"], file_name, output_dir)
        else:
            print("Assigning speaker labels...")
            diarize_model = model.load_diarize(device)
            diarize_segments = diarize_model(audio)
            output = whisperx.assign_word_speakers(diarize_segments, output)
            output_path = save_to_txt(output["segments"], file_name, output_dir)
        progress.update(1)  # 5

        status, output_data = test_and_finalize(output_path, start)
        print("Done.")

        return status, output_data

    except Exception as e:
        raise gr.Error(str(e))


def test_and_finalize(output_path: str, start: datetime):
    with open(output_path, "r", encoding="utf-8") as file:
        output_data = file.read()
    elapsed = (datetime.now() - start).total_seconds()
    status = f"Finished in {format_to_minutes(elapsed)}\nOutput saved to {output_path}"

    return status, output_data
