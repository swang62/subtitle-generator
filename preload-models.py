import torchaudio
import faster_whisper
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.constants import DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF

# Reference
# https://github.com/m-bain/whisperX/blob/v3.1.1/whisperx/alignment.py#L21

# See constants.py for available models/languages
models = ["large-v3-turbo"]
alignment_languages = ["en"]

for model in models:
    print(f"Loading model:{model}...")
    model = faster_whisper.WhisperModel(model)

for language in alignment_languages:
    print(f"Loading alignment:{language}...")

    model_name = ""
    if language in DEFAULT_ALIGN_MODELS_TORCH:
        model_name = DEFAULT_ALIGN_MODELS_TORCH[language]
    elif language in DEFAULT_ALIGN_MODELS_HF:
        model_name = DEFAULT_ALIGN_MODELS_HF[language]
    else:
        raise ValueError(f"No align-model for language: {language}")

    if model_name in torchaudio.pipelines.__all__:
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model()
        labels = bundle.get_labels()
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        except Exception as e:
            print(str(e))
            raise ValueError(f"Failed to load {model_name}")
        labels = processor.tokenizer.get_vocab()  # type: ignore

print("Loading voice activity detector...")
torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
    trust_repo=True,
)
print("Done.")
