import click
import gradio as gr
import uvicorn
from fastapi import FastAPI

from ui.config import MEDIA_FOLDER
from ui.main import demo

app = FastAPI()


# Backend API
@app.get("/api")
def api_route():
    return {"message": "This is your main app"}


# Frontend UI
app = gr.mount_gradio_app(app, demo, path="/", allowed_paths=[MEDIA_FOLDER])


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
