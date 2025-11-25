import os
from time import sleep

import gradio as gr
import torch
from pymediainfo import MediaInfo
from tqdm import tqdm

from ui.config import HF_TOKEN, MEDIA_FOLDER
from ui.constants import LANGUAGE_OPTIONS, MODELS
from ui.engine import generate_subtitles
from ui.utils import format_time_for_txt, is_valid_multimedia_file

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
    # Session state
    stored_data = gr.State("")
    stored_path = gr.State("")
    stored_speakers = gr.State("")

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
                    label="Selected filename",
                    info="Estimated length: 0",
                    interactive=False,
                )
                dir_selected = gr.Textbox(
                    label="File directory",
                    info="Resulting output will be saved to the same folder",
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
                    choices=list(LANGUAGE_OPTIONS.items()),  # type: ignore
                    label="Output language",
                    value="en",
                )

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=list(zip(MODELS.values(), MODELS.keys())),
                        label="Model",
                        value="base",
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
                detected_speakers = gr.Textbox(
                    info="Rename the default tags, make sure to seperate names with commas",
                    submit_btn="Rename",
                    label="Speaker names",
                    visible=False,
                )
                output_data = gr.TextArea(
                    label="Output",
                    interactive=False,
                    show_copy_button=True,
                    visible=True,
                )
                status = gr.Textbox(
                    show_label=False,
                    value=f"[Engine: {device.upper()}] Ready to process",
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

                result = generate_subtitles(progress=pbar, **settings)
                status = f"[Elapsed time: {format_time_for_txt(result['duration'])}] Saved output file."

                return (
                    gr.update(interactive=True),
                    status,
                    result["output_data"],
                    result["output_path"],
                    result["unique_speakers"],
                )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                return (
                    gr.update(interactive=True),
                    gr.update(value="Failed!"),
                    gr.Error(error_msg),
                    gr.update(value=""),
                    gr.update(value=""),
                )

    # Handlers
    transcribe_button.click(
        fn=start_process,
        show_progress_on=output_data,
        inputs=[
            file_selected,
            dir_selected,
            language_dropdown,
            model_dropdown,
            chunk_slider,
            mode_selection,
        ],
        outputs=[transcribe_button, status, stored_data, stored_path, stored_speakers],
    )
    transcribe_button.click(
        fn=lambda: [gr.update(value="Transcribing ..."), gr.update(interactive=False)],
        inputs=None,
        outputs=[status, transcribe_button],
    )

    def update_on_file_select(file_path: str):
        print(f"Selected file: {file_path}")
        media_duration = 0
        if not file_path or "." not in file_path:
            return gr.update(value="", info=f"Estimated length: {media_duration}"), ""

        track = MediaInfo.parse(file_path).tracks[0].to_data()
        media_duration = format_time_for_txt(track.get("duration", 0) / 1000)
        dir_path, file_name = os.path.split(file_path)
        return (
            gr.update(
                value=file_name,
                info=f"Estimated length: {media_duration}",
            ),
            dir_path,
        )

    file_input.change(
        fn=update_on_file_select,
        inputs=[file_input],
        outputs=[file_selected, dir_selected],
    )

    def update_on_mode_selection(mode: str):
        print(f"Selected mode: {mode}")
        return "👉 Transcribe" if mode == "transcribe" else "👉 Generate"

    mode_selection.change(
        fn=update_on_mode_selection,
        inputs=[mode_selection],
        outputs=[transcribe_button],
    )

    def update_speakers(
        stored_data: str, stored_path: str, stored_speakers: str, update_speakers: str
    ):
        old_speakers = stored_speakers.split(",")
        new_speakers = update_speakers.split(",")

        new_data = stored_data
        for old, new in zip(old_speakers, new_speakers):
            new_data = new_data.replace(old, new)

        with open(stored_path, "w", encoding="utf-8") as file:
            print(f"Replacing speakers in {stored_path}...")
            file.write(new_data)

        return new_data

    detected_speakers.submit(
        fn=update_speakers,
        inputs=[stored_data, stored_path, stored_speakers, detected_speakers],
        outputs=[output_data],
    )

    # Global state triggers
    stored_data.change(
        fn=lambda data: gr.update(value=data),
        inputs=[stored_data],
        outputs=[output_data],
    )
    stored_speakers.change(
        fn=lambda speakers, mode: gr.update(visible=True, placeholder=speakers)
        if mode == "transcribe"
        else gr.update(visible=False),
        inputs=[stored_speakers, mode_selection],
        outputs=[detected_speakers],
    )

if "__name__" == "__main__":
    demo.launch(allowed_paths=[MEDIA_FOLDER])
