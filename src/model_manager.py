import whisperx


class ModelCache:
    """Manages loading and caching of WhisperX models."""

    def __init__(self):
        # Data
        self._current_model = None
        self._current_align_model = None
        self._current_align_metadata = None
        self._current_audio_data = None

        # Settings
        self._current_model_name = None
        self._current_language_code = None
        self._current_audio_file = None
        self._current_device = None

    def load_model(self, model_name: str, device: str):
        if (
            self._current_model is None
            or model_name != self._current_model_name
            or device != self._current_device
        ):
            print(f"Loading model:{model_name} on device:{device}")
            compute_type = "float16" if device == "cuda" else "float32"

            self._current_model = whisperx.load_model(
                model_name, device=device, compute_type=compute_type
            )
            self._current_model_name = model_name
            self._current_device = device

        return self._current_model

    def load_align_model(self, language_code: str, device: str):
        if (
            self._current_align_model is None
            or language_code != self._current_language_code
            or device != self._current_device
        ):
            print(f"Loading alignment model:{language_code} on device:{device}")
            self._current_align_model, self._current_align_metadata = (
                whisperx.load_align_model(language_code=language_code, device=device)
            )
            self._current_language_code = language_code
            self._current_device = device

        return self._current_align_model, self._current_align_metadata

    def load_audio(self, audio_file: str):
        if self._current_audio_data is None or audio_file != self._current_audio_file:
            print(f"Loading audio file:{audio_file}")
            self._current_audio_data = whisperx.load_audio(audio_file)
            self._current_audio_file = audio_file

        return self._current_audio_data


cache = ModelCache()
