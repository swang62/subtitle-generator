import os
import mimetypes
import re

from src.constants import SUPPORTED_EXTENSIONS


def is_valid_multimedia_file(output_dir: str, file_name: str):
    """Checks if the file path corresponds to a real video/audio file"""

    file_path = os.path.join(output_dir, file_name)
    normalized_path = os.path.normpath(file_path)
    mime_type, _ = mimetypes.guess_type(normalized_path)
    is_supported_mime = mime_type and (
        mime_type.startswith("audio") or mime_type.startswith("video")
    )

    return is_supported_mime or normalized_path.lower().endswith(SUPPORTED_EXTENSIONS)


def format_to_minutes(elapsed: float):
    """Formats seconds to readable string in minutes"""

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # 0m 15s
    return f"{minutes}m {seconds}s"


def format_time(time_in_seconds):
    """Formats time in seconds to a readable time format."""

    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int(time_in_seconds % 60)
    milliseconds = int((time_in_seconds - int(time_in_seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def save_to_srt(segments, file_name: str, output_dir: str):
    """Formats and saves SRT with same name as the original video"""

    srt_file_name = re.sub(r"\.\w+$", ".srt", file_name)
    file_path = os.path.join(output_dir, srt_file_name)

    print(f"Saving to {file_path}...")
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(
            f"{i}\n{format_time(segment['start'])} --> {format_time(segment['end'])}\n{segment['text'].strip()}\n\n"
            for i, segment in enumerate(segments, 1)
        )

    return file_path
