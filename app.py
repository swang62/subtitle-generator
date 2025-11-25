from typing import Optional

import click
import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile

from api.engine import asr
from ui.config import MEDIA_FOLDER
from ui.constants import LANGUAGE_OPTIONS
from ui.main import demo

app = FastAPI(
    title="whisperx-asr", swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    return await asr(
        audio_file=audio_file,
        encode=encode,
        language=language,
        initial_prompt=initial_prompt,
        diarize=enable_diarization,
        return_speaker_embeddings=return_speaker_embeddings,
        output=output,
    )


# Frontend UI
app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    allowed_paths=[MEDIA_FOLDER],
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
