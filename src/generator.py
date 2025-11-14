import os
import whisperx
import gradio as gr
from src.model_manager import cache
from src.utils import format_to_minutes, save_to_srt
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
        audio = model.load_audio(file_path)
        progress.update(1)  # 2

        # Transcribe or translate
        print("Transcribing...")
        output = model.transcribe(audio, **options)  # type: ignore
        progress.update(1)  # 3

        # Make sure alignment is possible
        input_language = output.get("language")
        output_language = language or input_language
        if input_language == output_language:
            print("Aligning segments...")

            align_model, align_metadata = cache.load_align_model(input_language, device)
            aligned = whisperx.align(
                output["segments"], align_model, align_metadata, audio, device
            )

            segments = aligned["segments"]
        else:
            print("Skipping alignment, language mismatch...")
            segments = output["segments"]
        progress.update(1)  # 4

        # Save file and read output
        output_path = save_to_srt(segments, file_name, output_dir)
        progress.update(1)  # 5

        with open(output_path, "r", encoding="utf-8") as file:
            output_data = file.read()

        elapsed = (datetime.now() - start).total_seconds()
        total_time = (
            f"Finished in {format_to_minutes(elapsed)}\n[File saved to {output_path}]"
        )

        print("Done.")
        return total_time, output_data, output_path

    except Exception as e:
        raise gr.Error(str(e))
