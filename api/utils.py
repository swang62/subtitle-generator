import json
import os
from typing import BinaryIO, TextIO

import ffmpeg
import numpy as np
from faster_whisper.utils import format_timestamp


class ResultWriter:
    extension: str

    def __init__(self):
        self.output_dir = "./"

    def __call__(self, result: dict, audio_path: str):
        audio_basename = os.path.basename(audio_path)
        output_path = os.path.join(
            self.output_dir, audio_basename + "." + self.extension
        )

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f)

    def write_result(self, result: dict, file: TextIO):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result: dict, file: TextIO):
        for segment in result["segments"]:
            print(segment.text.strip(), file=file, flush=True)


class WriteSRT(ResultWriter):
    extension: str = "srt"

    def write_result(self, result: dict, file: TextIO):
        for i, segment in enumerate(result["segments"], start=1):
            # write srt lines
            print(
                f"{i}\n"
                f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
                f"{segment.text.strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: TextIO):
        json.dump(result, file, ensure_ascii=False)


def load_audio(file: BinaryIO, encode=True):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    """
    if encode:
        print("Encoding audio...")
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
                .run(
                    cmd="ffmpeg",
                    capture_stdout=True,
                    capture_stderr=True,
                    input=file.read(),
                )
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def write_result(result: dict, file: TextIO, output: str | None):
    if output == "srt":
        WriteSRT().write_result(result, file=file)
    elif output == "json":
        WriteJSON().write_result(result, file=file)
    else:
        WriteTXT().write_result(result, file=file)
