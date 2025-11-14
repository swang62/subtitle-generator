import os
import gradio as gr
import torch
from tqdm import tqdm
from src.config import HF_TOKEN, MEDIA_FOLDER
from src.constants import LANGUAGE_OPTIONS, MODELS
from src.generator import generate_subtitles
from src.utils import is_valid_multimedia_file
from time import sleep

device = "cuda" if torch.cuda.is_available() else "cpu"

css = """
.status textarea 
{
    text-align: center; 
    display: block
}
.choices 
{
    justify-content: center;
    display: flex
}
"""

with gr.Blocks(theme=gr.themes.Ocean(), css=css, title="Subtitle Generator") as demo:  # type: ignore
    # Header
    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown("# 🎙️ Subtitle Generator")

    # Main content
    with gr.Row():
        # Left column
        with gr.Column(variant="panel"):
            gr.Markdown("### 📁 Input")
            with gr.Group():
                file_selected = gr.Textbox(
                    label="Selected file",
                    interactive=False,
                )
                dir_selected = gr.Textbox(
                    label="Output location",
                    interactive=False,
                )
                file_input = gr.FileExplorer(
                    label="File browser",
                    root_dir=MEDIA_FOLDER,
                    file_count="single",
                )

        # Right column
        with gr.Column(
            variant="panel",
        ):
            gr.Markdown("### ⚙️ Run")
            with gr.Group():
                language_dropdown = gr.Dropdown(
                    choices=list(
                        zip(LANGUAGE_OPTIONS.keys(), LANGUAGE_OPTIONS.values())
                    ),
                    label="Output language",
                    value="en",
                )

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=list(zip(MODELS.values(), MODELS.keys())),
                        label="Model",
                        value="large-v3-turbo",
                    )
                    chunk_slider = gr.Slider(
                        info="Lower sizes are better for asian languages",
                        minimum=8,
                        maximum=30,
                        step=1,
                        label="Chunk size",
                        value=30,
                    )
                mode_selection = gr.Radio(
                    elem_classes="choices",
                    choices=[
                        ("Generate subtitles (.srt)", "generate"),
                        (
                            "Transcribe meeting (with speakers)"
                            if HF_TOKEN
                            else "Transcribe meeting (requires token)",
                            "transcribe",
                        ),
                    ],
                    interactive=HF_TOKEN != "",
                    show_label=False,
                    value="generate",
                )
                transcribe_button = gr.Button("👉 Generate", variant="primary")

            gr.Markdown("### 📝 Results")
            with gr.Group():
                output_content = gr.TextArea(
                    show_label=False,
                    interactive=False,
                    show_copy_button=True,
                    visible=True,
                )
                status = gr.Textbox(
                    show_label=False,
                    scale=True,
                    value=f"Engine: {device.upper()}",
                    elem_classes="status",
                )

    # Function to process transcription
    def start_process(
        file_name,
        output_dir,
        language,
        model_name,
        chunk_size,
        mode,
        progress=gr.Progress(track_tqdm=True),
    ) -> tuple:
        with tqdm(total=5) as pbar:
            try:
                if not is_valid_multimedia_file(output_dir, file_name):
                    sleep(0.2)
                    raise ValueError("Please select a valid video or audio file.")

                device = "cuda" if torch.cuda.is_available() else "cpu"
                settings = {
                    "file_name": file_name,
                    "output_dir": output_dir,
                    "language": language,
                    "model_name": model_name,
                    "device": device,
                    "chunk_size": int(chunk_size),
                    "mode": mode,
                }
                print(settings)
                pbar.set_description("Processing...")

                status, content = generate_subtitles(progress=pbar, **settings)
                return status, content

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                return (
                    gr.update(value="Failed!"),
                    gr.Error(error_msg),
                    None,
                )

    # Handlers
    def update_on_file_select(file_path: str):
        print(f"Selected file: {file_path}")
        if not file_path:
            return "", ""

        dir_path, file_name = os.path.split(file_path)
        return file_name, dir_path

    file_input.change(
        fn=update_on_file_select,
        inputs=[file_input],
        outputs=[file_selected, dir_selected],
    )

    def update_on_mode_selection(mode: str):
        print(f"Selected mode: {mode}")

        if mode == "transcribe":
            return "👉 Transcribe"
        else:
            return "👉 Generate"

    mode_selection.change(
        fn=update_on_mode_selection,
        inputs=[mode_selection],
        outputs=[transcribe_button],
    )

    transcribe_button.click(
        fn=start_process,
        show_progress_on=output_content,
        inputs=[
            file_selected,
            dir_selected,
            language_dropdown,
            model_dropdown,
            chunk_slider,
            mode_selection,
        ],
        outputs=[status, output_content],
    )

    transcribe_button.click(
        fn=lambda lang: gr.update(value=f"Transcribing to language: {lang} ..."),
        inputs=[language_dropdown],
        outputs=[status],
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=[MEDIA_FOLDER])
