from __future__ import annotations


LONG_REFERENCE_TIP_MARKDOWN = (
    "**Long-reference tip:** Upload multiple clean, shorter clips from the same "
    "speaker and arrange them in the desired order. This matches v4-Small "
    "training. A single uninterrupted long recording is accepted but has not "
    "been evaluated."
)


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
            if candidate is not None:
                text = str(candidate).strip()
                if text:
                    return text
        return None
    for attribute in ("path", "name"):
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            text = str(candidate).strip()
            if text:
                return text
    return None


def resolve_gradio_reference_wavs(uploaded_audio: object) -> list[str]:
    """Normalize single or multiple Gradio audio uploads to file paths."""

    if uploaded_audio is None:
        return []
    values = uploaded_audio if isinstance(uploaded_audio, (list, tuple)) else [uploaded_audio]
    paths = [coerce_gradio_file_path(value) for value in values]
    return [path for path in paths if path is not None]
