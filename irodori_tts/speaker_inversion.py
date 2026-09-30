from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file


SPEAKER_INVERSION_UNCOND_MODES = {"mask", "noise"}
SPEAKER_INVERSION_SAFETENSORS_SUFFIX = ".speaker.safetensors"
SPEAKER_INVERSION_BASE_SAFETENSORS_SUFFIX = ".speaker-base.safetensors"
SPEAKER_EMBEDDING_KEY = "speaker_embedding"
SPEAKER_PRE_NORM_EMBEDDING_KEY = "speaker_pre_norm_embedding"
SPEAKER_INVERSION_BASE_FORMAT_VERSION = "2"


def speaker_inversion_checkpoint_sha256(path: str | Path) -> str:
    """
    Speaker Inversion の基準状態を生成した checkpoint の SHA-256 を計算する。

    Args:
        path (str | Path): ハッシュを計算する checkpoint のパス

    Returns:
        str: 16進表記の SHA-256
    """

    source = Path(path).expanduser()
    hasher = hashlib.sha256()
    # Checkpoint paths are portable, but their bytes define the encoder that produced the base.
    with source.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_speaker_embedding_tensor(
    tensor: torch.Tensor,
    *,
    speaker_dim: int,
    field_name: str = SPEAKER_EMBEDDING_KEY,
) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"{field_name} must have shape (tokens, dim), got {tuple(tensor.shape)}")
    if int(tensor.shape[0]) <= 0:
        raise ValueError(f"{field_name} must contain at least one token.")
    if int(tensor.shape[1]) != int(speaker_dim):
        raise ValueError(
            f"{field_name} dim mismatch: expected {int(speaker_dim)}, got {int(tensor.shape[1])}"
        )

    return tensor.detach().float().contiguous()


def speaker_token_limit(
    *,
    ref_max_seconds: float,
    frames_per_second: float,
    latent_patch_size: int,
    speaker_patch_size: int,
) -> int | None:
    """
    話者条件として受け付けられる最大トークン数を返す。

    Args:
        ref_max_seconds (float): 参照音声の上限秒数 (0 以下は上限なし)
        frames_per_second (float): パッチ化前の潜在表現の 1 秒あたりのフレーム数
        latent_patch_size (int): 潜在表現のパッチサイズ
        speaker_patch_size (int): Speaker Encoder のパッチサイズ

    Returns:
        int | None: 参照音声の最大パッチ数に、先頭へ追加される平均トークン 1 個を足した数 (上限なしなら None)
    """

    if ref_max_seconds <= 0:
        return None
    # 参照音声の処理と同じく、フレーム数を切り上げてから2段のパッチ化でそれぞれ端数を捨てる
    latent_frames = math.ceil(ref_max_seconds * frames_per_second)
    latent_patches = latent_frames // int(latent_patch_size)
    return latent_patches // int(speaker_patch_size) + 1


def is_speaker_inversion_safetensors_path(path: str | Path) -> bool:
    return Path(path).name.endswith(SPEAKER_INVERSION_SAFETENSORS_SUFFIX)


class SpeakerInversionEmbedding(nn.Module):
    """Learned speaker/style tokens that bypass the reference latent speaker encoder."""

    def __init__(
        self,
        *,
        num_tokens: int,
        speaker_dim: int,
        init_std: float,
        init_embedding: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        num_tokens = int(num_tokens)
        speaker_dim = int(speaker_dim)
        init_std = float(init_std)
        if num_tokens <= 0:
            raise ValueError(f"speaker inversion tokens must be > 0, got {num_tokens}")
        if speaker_dim <= 0:
            raise ValueError(f"speaker_dim must be > 0, got {speaker_dim}")
        if init_std < 0:
            raise ValueError(f"speaker inversion init_std must be >= 0, got {init_std}")

        if init_embedding is None:
            embedding = torch.randn(num_tokens, speaker_dim, dtype=torch.float32) * init_std
        else:
            embedding = normalize_speaker_embedding_tensor(
                init_embedding,
                speaker_dim=speaker_dim,
                field_name=SPEAKER_EMBEDDING_KEY,
            )
            if int(embedding.shape[0]) != num_tokens:
                raise ValueError(
                    "speaker inversion init embedding token mismatch: "
                    f"expected {num_tokens}, got {int(embedding.shape[0])}"
                )
        self.embedding = nn.Parameter(embedding)

    @property
    def num_tokens(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def speaker_dim(self) -> int:
        return int(self.embedding.shape[1])

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.embedding.to(device=device, dtype=dtype)[None, :, :].expand(
            int(batch_size),
            -1,
            -1,
        )
        mask = torch.ones((int(batch_size), self.num_tokens), dtype=torch.bool, device=device)
        return state, mask


def _extract_embedding_payload(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if SPEAKER_EMBEDDING_KEY in raw:
        embedding = raw[SPEAKER_EMBEDDING_KEY]
        return {SPEAKER_EMBEDDING_KEY: embedding}

    raise ValueError(f"Speaker inversion file is missing '{SPEAKER_EMBEDDING_KEY}'.")


def normalize_speaker_inversion_payload(
    raw: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    payload = _extract_embedding_payload(raw)
    embedding = payload[SPEAKER_EMBEDDING_KEY]

    out: dict[str, torch.Tensor] = {
        SPEAKER_EMBEDDING_KEY: embedding,
    }

    return out


def load_speaker_inversion_payload(
    path: str | Path,
) -> dict[str, torch.Tensor]:
    source = Path(path).expanduser()
    if not is_speaker_inversion_safetensors_path(source):
        raise ValueError(
            "Speaker Inversion embeddings must use the "
            f"{SPEAKER_INVERSION_SAFETENSORS_SUFFIX!r} suffix: {source}"
        )
    raw = load_safetensors_file(source, device="cpu")

    out = normalize_speaker_inversion_payload(raw)
    return out


def load_speaker_inversion_base_payload(
    path: str | Path,
    *,
    expected_checkpoint: str | Path | None = None,
    expected_speaker_dim: int | None = None,
    expected_speaker_patch_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Load a pre-normalization speaker state extracted from an ordinary reference."""

    source = Path(path).expanduser()
    if not source.name.endswith(SPEAKER_INVERSION_BASE_SAFETENSORS_SUFFIX):
        raise ValueError(
            "Speaker Inversion base embeddings must use the "
            f"{SPEAKER_INVERSION_BASE_SAFETENSORS_SUFFIX!r} suffix: {source}"
        )
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format_version") != SPEAKER_INVERSION_BASE_FORMAT_VERSION:
        raise ValueError(
            "Speaker Inversion base file has an unsupported or missing format_version."
        )
    if expected_checkpoint is not None:
        expected_checkpoint_path = Path(expected_checkpoint).expanduser().resolve()
        actual_fingerprint = metadata.get("checkpoint_sha256")
        expected_fingerprint = speaker_inversion_checkpoint_sha256(expected_checkpoint_path)
        if actual_fingerprint != expected_fingerprint:
            raise ValueError(
                "Speaker Inversion base checkpoint mismatch: "
                f"expected SHA-256 {expected_fingerprint}, got {actual_fingerprint!r}."
            )
    if expected_speaker_dim is not None and metadata.get("speaker_dim") != str(
        int(expected_speaker_dim)
    ):
        raise ValueError(
            "Speaker Inversion base speaker_dim mismatch: "
            f"expected {int(expected_speaker_dim)}, got {metadata.get('speaker_dim')!r}."
        )
    if expected_speaker_patch_size is not None and metadata.get("speaker_patch_size") != str(
        int(expected_speaker_patch_size)
    ):
        raise ValueError(
            "Speaker Inversion base speaker_patch_size mismatch: "
            f"expected {int(expected_speaker_patch_size)}, "
            f"got {metadata.get('speaker_patch_size')!r}."
        )

    raw = load_safetensors_file(source, device="cpu")
    if SPEAKER_PRE_NORM_EMBEDDING_KEY not in raw:
        raise ValueError(
            f"Speaker Inversion base file is missing '{SPEAKER_PRE_NORM_EMBEDDING_KEY}'."
        )
    raw_embedding = raw[SPEAKER_PRE_NORM_EMBEDDING_KEY]
    if raw_embedding.ndim not in {2, 3}:
        raise ValueError(
            f"{SPEAKER_PRE_NORM_EMBEDDING_KEY} must be a rank-2 or rank-3 tensor, "
            f"got {tuple(raw_embedding.shape)}"
        )
    return {
        SPEAKER_PRE_NORM_EMBEDDING_KEY: normalize_speaker_embedding_tensor(
            raw_embedding,
            speaker_dim=(
                int(raw_embedding.shape[-1])
                if expected_speaker_dim is None
                else int(expected_speaker_dim)
            ),
            field_name=SPEAKER_PRE_NORM_EMBEDDING_KEY,
        )
    }


def save_speaker_inversion_base_safetensors(
    path: str | Path,
    pre_norm_embedding: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    metadata: dict[str, str] | None = None,
) -> None:
    """Save reference-derived local speaker tokens taken before speaker normalization."""

    target = Path(path).expanduser()
    if not target.name.endswith(SPEAKER_INVERSION_BASE_SAFETENSORS_SUFFIX):
        raise ValueError(
            "Speaker Inversion base output must use the "
            f"{SPEAKER_INVERSION_BASE_SAFETENSORS_SUFFIX!r} suffix: {target}"
        )
    if pre_norm_embedding.ndim not in {2, 3}:
        raise ValueError(
            f"{SPEAKER_PRE_NORM_EMBEDDING_KEY} must be a rank-2 or rank-3 tensor, "
            f"got {tuple(pre_norm_embedding.shape)}"
        )
    normalized = normalize_speaker_embedding_tensor(
        pre_norm_embedding,
        speaker_dim=int(pre_norm_embedding.shape[-1]),
        field_name=SPEAKER_PRE_NORM_EMBEDDING_KEY,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    file_metadata = {
        "format_version": SPEAKER_INVERSION_BASE_FORMAT_VERSION,
        "local_tokens": str(int(normalized.shape[0])),
        "speaker_dim": str(int(normalized.shape[1])),
    }
    if metadata is not None:
        file_metadata.update(metadata)
    save_safetensors_file(
        {SPEAKER_PRE_NORM_EMBEDDING_KEY: normalized.to(dtype=dtype)},
        str(target),
        metadata=file_metadata,
    )


def save_speaker_inversion_safetensors(
    path: str | Path,
    payload: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
) -> None:
    target = Path(path)
    if not is_speaker_inversion_safetensors_path(target):
        raise ValueError(
            "Speaker Inversion safetensors output must use the "
            f"{SPEAKER_INVERSION_SAFETENSORS_SUFFIX!r} suffix: {target}"
        )
    normalized = normalize_speaker_inversion_payload(payload)
    tensors = {
        SPEAKER_EMBEDDING_KEY: normalized[SPEAKER_EMBEDDING_KEY].to(dtype=dtype),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors_file(tensors, str(target), metadata={})


def speaker_inversion_batch_tensors(
    speaker_embedding: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding = speaker_embedding.to(device=device, dtype=dtype)
    state = embedding[None, :, :].expand(int(batch_size), -1, -1)
    mask = torch.ones((int(batch_size), embedding.shape[0]), dtype=torch.bool, device=device)
    return state, mask


def speaker_inversion_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    module = getattr(model, "speaker_inversion", None)
    if not isinstance(module, SpeakerInversionEmbedding):
        raise ValueError("Model does not have an enabled SpeakerInversionEmbedding module.")

    return {
        SPEAKER_EMBEDDING_KEY: module.embedding.detach().cpu().float().clone(),
    }


def save_speaker_inversion_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = speaker_inversion_state_dict(model)
    save_speaker_inversion_safetensors(path, state)
