from __future__ import annotations


class GradioReferenceFiles:
    """
    Gradio の参照音声アップロード値を推論用のパスへ変換する。
    """

    LONG_REFERENCE_TIP_MARKDOWN = (
        "**Long-reference tip:** Upload multiple clean, shorter clips from the same "
        "speaker and arrange them in the desired order. This matches v4-Small "
        "training. A single uninterrupted long recording is accepted but has not "
        "been evaluated."
    )

    @staticmethod
    def resolve_reference_wavs(uploaded_audio: object) -> list[str]:
        """
        Gradio の単一または複数の参照音声を順序付きパスへ変換する。

        Args:
            uploaded_audio (object): Gradio が返す単一または複数のアップロード値

        Returns:
            list[str]: 有効な参照音声パス
        """

        if uploaded_audio is None:
            return []
        values = uploaded_audio if isinstance(uploaded_audio, (list, tuple)) else [uploaded_audio]
        paths = [GradioReferenceFiles.coerce_file_path(value) for value in values]
        return [path for path in paths if path is not None]

    @staticmethod
    def filter_reference_wavs(
        reference_wavs: list[str],
        *,
        supports_speaker_condition: bool,
    ) -> tuple[list[str], str | None]:
        """
        checkpoint が利用できる参照音声だけを残す。

        Args:
            reference_wavs (list[str]): アップロード順を保持した参照音声パス
            supports_speaker_condition (bool): checkpoint が話者条件を利用できるか

        Returns:
            tuple[list[str], str | None]: 利用する参照音声と、無視した場合の通知文
        """

        if reference_wavs and supports_speaker_condition is False:
            return [], (
                "uploaded reference audio was ignored because this checkpoint does not support "
                "speaker conditioning."
            )
        return reference_wavs, None

    @staticmethod
    def coerce_file_path(value: object) -> str | None:
        """
        Gradio のアップロード値から空でないファイルパスを取得する。

        Args:
            value (object): Gradio が返す文字列、辞書、またはファイルオブジェクト

        Returns:
            str | None: 前後の空白を除いたパス。パスを取得できない場合は None
        """

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
