import os

from dotenv import load_dotenv

load_dotenv()

MEDIA_FOLDER = os.getenv("MEDIA_FOLDER", "/media")
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "base")
