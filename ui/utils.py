import mimetypes
import os
import re
from typing import Any

from ui.constants import SUPPORTED_EXTENSIONS


def is_valid_multimedia_file(output_dir: str, file_name: str):
    """Checks if the file path corresponds to a real video/audio file"""

    file_path = os.path.join(output_dir, file_name)
    normalized_path = os.path.normpath(file_path)
    mime_type, _ = mimetypes.guess_type(normalized_path)
    is_supported_mime = mime_type and (
        mime_type.startswith("audio") or mime_type.startswith("video")
    )

    return is_supported_mime or normalized_path.lower().endswith(SUPPORTED_EXTENSIONS)


def format_time_for_txt(duration: float):
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    # 00:00:15
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_time_for_srt(duration: float):
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    milliseconds = int((duration - int(duration)) * 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def save_to_file(
    segments: list[dict[str, Any]], file_name: str, output_dir: str, output_format: str
):
    srt_file_name = re.sub(r"\.\w+$", f".{output_format}", file_name)
    file_path = os.path.join(output_dir, srt_file_name)

    print(f"Saving to {file_path}...")
    with open(file_path, "w", encoding="utf-8") as file:
        if output_format == "srt":
            for i, segment in enumerate(segments, 1):
                start = format_time_for_srt(segment["start"])
                end = format_time_for_srt(segment["end"])
                text = segment["text"].strip()
                file.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        else:
            for segment in segments:
                start = format_time_for_txt(segment["start"])
                speaker = segment["speaker"].strip()
                text = segment["text"].strip()
                file.write(f"[{start}] {speaker}: {text}\n")

    return file_path
