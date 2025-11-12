import gradio as gr
import torch
from tqdm import tqdm
from src.config import MEDIA_FOLDER
from src.constants import LANGUAGE_OPTIONS, MODELS
from src.generator import generate_subtitles
from src.utils import is_valid_file

with gr.Blocks(theme=gr.themes.Ocean()) as demo:  # type: ignore
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
            transcribe_button = gr.Button("👉 Transcribe", variant="primary")

            gr.Markdown("### 📝 Results")
            with gr.Group():
                output_content = gr.TextArea(
                    label="Subtitles",
                    interactive=False,
                    show_copy_button=True,
                    visible=True,
                )
                output_path = gr.DownloadButton("Download")

    # Function to process transcription
    def start_process(
        file_name,
        output_dir,
        language,
        model_name,
        chunk_size,
        progress=gr.Progress(track_tqdm=True),
    ):
        with tqdm(total=5) as pbar:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                settings = {
                    "file_name": file_name,
                    "output_dir": output_dir,
                    "language": language,
                    "model_name": model_name,
                    "device": device,
                    "chunk_size": int(chunk_size),
                }
                print(settings)

                content, path = generate_subtitles(progress=pbar, **settings)
                return content, path

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return [gr.update(value=error_msg)] * 2

    # Handlers
    def update_on_file_select(file: str):
        if not file:
            return "", ""
        path = file.split("\\")
        filename = path[-1]
        dirpath = "\\".join(path[:-1])
        return filename, dirpath

    file_input.change(
        fn=update_on_file_select,
        inputs=[file_input],
        outputs=[file_selected, dir_selected],
    )

    def validate_input(file_name, output_dir, *args) -> list[dict]:
        return [
            gr.validate(
                is_valid_file(output_dir + "\\" + file_name),
                "Invalid file.",
            )
        ] + [gr.validate(True, "")] * 4

    transcribe_button.click(
        fn=start_process,
        validator=validate_input,
        inputs=[
            file_selected,
            dir_selected,
            language_dropdown,
            model_dropdown,
            chunk_slider,
        ],
        outputs=[output_content, output_path],
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=[MEDIA_FOLDER])
