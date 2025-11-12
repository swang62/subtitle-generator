import whisperx


class ModelManager:
    """Manages loading and caching of WhisperX models."""

    def __init__(self):
        self._current_model = None
        self._current_model_name = None
        self._current_device = None

    def _needs_load(self, model_name, device):
        """Checks if a new model needs to be loaded."""
        return (
            self._current_model is None
            or model_name != self._current_model_name
            or device != self._current_device
        )

    def load_model(self, model_name, device):
        """Loads a model if necessary or returns the cached model."""
        if self._needs_load(model_name, device):
            print(f"Using model:{model_name} on device:{device}")
            compute_type = "float16" if device == "cuda" else "float32"

            self._current_model = whisperx.load_model(
                model_name, device=device, compute_type=compute_type
            )
            self._current_model_name = model_name
            self._current_device = device

        return self._current_model


manager = ModelManager()
