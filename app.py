from typing import Optional
from urllib.parse import quote

import click
import gradio as gr
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import StreamingResponse

from api.engine import asr
from shared.config import MEDIA_FOLDER
from shared.constants import LANGUAGE_OPTIONS
from ui.main import CSS, demo

app = FastAPI(title="whisperx-asr")


# Backend API
@app.post("/asr")
async def asr_endpoint(
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True),
    language: Optional[str] = Query(default=None, enum=list(LANGUAGE_OPTIONS.values())),
    initial_prompt: Optional[str] = Query(default=None),
    enable_diarization: bool = Query(default=True),
    return_speaker_embeddings: bool = Query(default=True),
    output: Optional[str] = Query(default="txt", enum=["txt", "srt", "json"]),
):
    output_file = asr(
        audio_file=audio_file,
        encode=encode,
        language=language,
        initial_prompt=initial_prompt,
        diarize=enable_diarization,
        return_speaker_embeddings=return_speaker_embeddings,
        output=output,
    )

    return StreamingResponse(
        output_file,
        media_type="text/plain",
        headers={
            "Asr-Engine": "whisperx",
            "Content-Disposition": f'attachment; filename="{quote(str(audio_file.filename))}.{output}"',
        },
    )


# Frontend UI
app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    allowed_paths=[MEDIA_FOLDER],
    theme=gr.themes.Ocean(),  # type: ignore
    css=CSS,
)


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    help="Host for entire webserver (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    default=7860,
    help="Port (default: 7860)",
)
def start(host: str, port: int):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
