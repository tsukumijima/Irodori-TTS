from __future__ import annotations


def coerce_gradio_file_path(value: object) -> str | None:
    """Normalize a Gradio upload value to a non-empty file path."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("path", "name"):
            candidate = value.get(key)
            if candidate is not None and str(candidate).strip():
                return str(candidate)
        return None
    candidate = getattr(value, "name", None)
    if candidate is not None and str(candidate).strip():
        return str(candidate)
    text = str(value).strip()
    return text or None


def resolve_gradio_reference_wavs(uploaded_audio: object) -> list[str]:
    """Normalize single or multiple Gradio audio uploads to file paths."""

    if uploaded_audio is None:
        return []
    values = uploaded_audio if isinstance(uploaded_audio, (list, tuple)) else [uploaded_audio]
    paths = [coerce_gradio_file_path(value) for value in values]
    return [path for path in paths if path is not None]
