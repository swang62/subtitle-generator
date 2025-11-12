import whisperx
import gradio as gr
from src.model_manager import manager
from src.utils import save_to_srt


# Main function to transcribe/translate audio
def generate_subtitles(
    file_name: str,
    output_dir: str,
    language: str | None,
    model_name: str,
    device: str,
    chunk_size: int,
    progress,
):
    file_path = output_dir + "\\" + file_name

    # Config
    options = {}
    if language:
        options["language"] = language
    options["chunk_size"] = chunk_size

    try:
        # Load whisper model
        model = manager.load_model(model_name, device)
        progress.update(1)  # 1

        # Load audio file
        print("Loading in audio...")
        audio = whisperx.load_audio(file_path)
        progress.update(1)  # 2

        # Transcribe or translate
        print("Transcribing...")
        output = model.transcribe(audio, **options)  # type: ignore
        progress.update(1)  # 3

        # Make sure alignment is valid
        input_language = str(output.get("language"))
        output_language = language or input_language
        if input_language == output_language:
            # Align segments/chunks with timestamps
            print("Aligning segments...")
            model_a, metadata = whisperx.load_align_model(
                language_code=input_language, device=device
            )
            aligned = whisperx.align(
                output["segments"], model_a, metadata, audio, device
            )
            segments = aligned["segments"]
        else:
            print("Skip alignment, language mismatch...")
            segments = output["segments"]
        progress.update(1)  # 4

        # Save file and read output
        output_path = save_to_srt(segments, file_name, output_dir)
        progress.update(1)  # 5

        with open(output_path, "r", encoding="utf-8") as file:
            output_data = file.read()
        print("Done.")

        return output_data, output_path

    except Exception as e:
        error_message = f"Error during transcription: {str(e)}"
        print(error_message)
        raise gr.Error(error_message)
