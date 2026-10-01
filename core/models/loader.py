from __future__ import annotations

import hashlib
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import whisper_s2t
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm.auto import tqdm

from config.constants import WHISPER_MODELS
from core.exceptions import ModelLoadError
from core.logging_config import get_logger
from core.models.metadata import ModelMetadata
from utils import get_optimal_cpu_threads

logger = get_logger(__name__)


class _NullWriter:
    def write(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass


def _ensure_streams() -> None:
    if sys.stdout is None:
        sys.stdout = _NullWriter()
    if sys.stderr is None:
        sys.stderr = _NullWriter()


class _ProgressTqdm(tqdm):
    def __init__(
        self,
        *args,
        progress_callback=None,
        completed_bytes=0,
        total_all_bytes=0,
        cancel_event=None,
        **kwargs,
    ):
        self._progress_callback = progress_callback
        self._completed_bytes = completed_bytes
        self._total_all_bytes = total_all_bytes
        self._cancel_event = cancel_event
        kwargs.pop("name", None)
        if "file" in kwargs and kwargs["file"] is None:
            kwargs["file"] = _NullWriter()
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise InterruptedError("Download cancelled")
        super().update(n)
        if self._progress_callback and self._total_all_bytes > 0:
            self._progress_callback(
                self._completed_bytes + int(self.n), self._total_all_bytes
            )


def _make_tqdm_class(callback, completed, total_all, cancel_event=None):
    class _BoundTqdm(_ProgressTqdm):
        def __init__(self, *args, **kwargs):
            kwargs["progress_callback"] = callback
            kwargs["completed_bytes"] = completed
            kwargs["total_all_bytes"] = total_all
            kwargs["cancel_event"] = cancel_event
            super().__init__(*args, **kwargs)

    return _BoundTqdm


def get_repo_id(model_name: str, precision: str) -> str:
    info = ModelMetadata.get_model_info(model_name, precision)
    if info is None:
        raise ModelLoadError(
            f"Unknown model/precision combination: {model_name} - {precision}"
        )
    return info["repo_id"]


_REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json")
_VOCABULARY_FILES = ("vocabulary.json", "vocabulary.txt")


class RepoFile(NamedTuple):
    name: str
    size: int
    blob_id: str
    sha256: Optional[str]


def _hub_cache_dir() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_cache_dir(repo_id: str) -> Path:
    return _hub_cache_dir() / ("models--" + repo_id.replace("/", "--"))


def _get_local_model_dir(repo_id: str) -> Path:
    return _hub_cache_dir() / "local_copies" / repo_id.replace("/", "--")


def _is_file_accessible(filepath: Path) -> bool:
    try:
        with open(filepath, "rb") as f:
            f.read(1)
        return True
    except (OSError, IOError):
        return False


def validate_model_path(path: str) -> bool:
    root = Path(path)
    return all(_is_file_accessible(root / name) for name in _REQUIRED_FILES) and any(
        _is_file_accessible(root / name) for name in _VOCABULARY_FILES
    )


def check_model_cached(repo_id: str) -> Optional[str]:
    snapshots = _repo_cache_dir(repo_id) / "snapshots"
    candidates = list(snapshots.iterdir()) if snapshots.is_dir() else []
    local_dir = _get_local_model_dir(repo_id)
    if local_dir.is_dir():
        candidates.append(local_dir)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if validate_model_path(str(candidate)):
            return str(candidate)
    return None


def get_repo_file_info(repo_id: str) -> tuple[str, list[RepoFile]]:
    info = HfApi().repo_info(repo_id, repo_type="model", files_metadata=True)
    files = [
        RepoFile(
            name=sibling.rfilename,
            size=sibling.size or 0,
            blob_id=sibling.blob_id,
            sha256=sibling.lfs.sha256 if sibling.lfs else None,
        )
        for sibling in info.siblings
    ]
    files.sort(key=lambda f: f.size)
    return info.sha, files


def _file_matches(path: Path, repo_file: RepoFile, exact: bool = True) -> bool:
    try:
        if not path.is_file() or path.stat().st_size != repo_file.size:
            return False
        if repo_file.sha256 and not exact:
            return True
        if repo_file.sha256:
            digest, expected = hashlib.sha256(), repo_file.sha256
        else:
            digest = hashlib.sha1(b"blob %d\0" % repo_file.size)
            expected = repo_file.blob_id
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == expected


def _reuse_identical_file(repo_id: str, revision: str, repo_file: RepoFile) -> bool:
    snapshots = _repo_cache_dir(repo_id) / "snapshots"
    if not snapshots.is_dir():
        return False
    target = snapshots / revision / repo_file.name
    for other in snapshots.iterdir():
        source = other / repo_file.name
        if other.name == revision or not _file_matches(source, repo_file):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source.resolve(), target)
        except OSError:
            shutil.copyfile(source, target)
        return True
    return False


def prepare_model_download(
    repo_id: str, revision: str, files: list[RepoFile]
) -> tuple[Optional[str], list[tuple[str, int]], Optional[Path]]:
    snapshot = _repo_cache_dir(repo_id) / "snapshots" / revision
    if all(_is_file_accessible(snapshot / f.name) for f in files) and validate_model_path(
        str(snapshot)
    ):
        return str(snapshot), [], None

    local_dir = _get_local_model_dir(repo_id)
    if validate_model_path(str(local_dir)):
        return None, [
            (f.name, 0 if _file_matches(local_dir / f.name, f, exact=False) else f.size)
            for f in files
        ], local_dir

    blobs = _repo_cache_dir(repo_id) / "blobs"
    to_download = []
    for f in files:
        if _is_file_accessible(snapshot / f.name):
            continue
        if (blobs / (f.sha256 or f.blob_id)).is_file():
            to_download.append((f.name, 0))
        elif not _reuse_identical_file(repo_id, revision, f):
            to_download.append((f.name, f.size))

    if not to_download and validate_model_path(str(snapshot)):
        return str(snapshot), [], None
    return None, to_download, None


def _on_rmtree_error(func, path, _exc_info):
    try:
        os.chmod(path, 0o777)
        func(path)
    except Exception:
        try:
            os.unlink(path)
        except Exception:
            pass


def _force_remove_tree(path: Path) -> None:
    try:
        entries = os.listdir(str(path))
    except OSError:
        return
    for entry in entries:
        entry_path = os.path.join(str(path), entry)
        try:
            os.unlink(entry_path)
        except OSError:
            _force_remove_tree(Path(entry_path))
    try:
        os.rmdir(str(path))
    except OSError:
        pass


def _get_repo_cache_path(repo_id: str) -> Optional[Path]:
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return Path(repo.repo_path)
    except Exception:
        pass

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        dir_name = "models--" + repo_id.replace("/", "--")
        candidate = Path(HF_HUB_CACHE) / dir_name
        if candidate.exists():
            return candidate
    except Exception:
        pass

    return None


def _clear_corrupted_cache(repo_id: str) -> None:
    target = _get_repo_cache_path(repo_id)
    if target is None:
        logger.warning(f"Could not locate cache directory for {repo_id}")
        return

    logger.info(f"Clearing entire model cache: {target}")
    shutil.rmtree(target, onerror=_on_rmtree_error)

    if target.exists():
        logger.warning("rmtree incomplete, forcing file-by-file removal")
        _force_remove_tree(target)

    if target.exists():
        logger.warning(
            f"Cache directory still exists after forced removal: {target}"
        )
    else:
        logger.info(f"Successfully cleared model cache: {target}")


def download_model_files(
    repo_id: str,
    revision: str,
    files_info: list[tuple[str, int]],
    local_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    _ensure_streams()

    total_bytes = sum(size for _, size in files_info)
    downloaded_bytes = 0

    for filename, size in files_info:
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Download cancelled")

        try:
            tqdm_cls = (
                _make_tqdm_class(
                    progress_callback, downloaded_bytes, total_bytes, cancel_event
                )
                if (progress_callback or cancel_event is not None)
                else None
            )
            dl_kwargs = {"repo_id": repo_id, "filename": filename, "revision": revision}
            if local_dir is not None:
                dl_kwargs["local_dir"] = str(local_dir)
            if tqdm_cls:
                dl_kwargs["tqdm_class"] = tqdm_cls
            hf_hub_download(**dl_kwargs)
        except InterruptedError:
            # Cancelled mid-file via the tqdm hook -- propagate as cancellation.
            raise
        except Exception as file_err:
            if cancel_event and cancel_event.is_set():
                # Failure was the cancellation; do NOT fall back to
                # snapshot_download (it would re-fetch the whole model).
                raise InterruptedError("Download cancelled") from file_err
            logger.warning(
                f"Per-file download failed for '{filename}': {file_err}. "
                f"Falling back to snapshot_download."
            )
            _ensure_streams()
            try:
                local_path = snapshot_download(
                    repo_id,
                    revision=revision,
                    local_dir=str(local_dir) if local_dir is not None else None,
                )
            except Exception as snap_err:
                raise snap_err from file_err
            if progress_callback:
                progress_callback(total_bytes, total_bytes)
            return local_path

        downloaded_bytes += size

        if progress_callback:
            progress_callback(downloaded_bytes, total_bytes)

    if local_dir is not None:
        local_path = str(local_dir)
    else:
        local_path = str(_repo_cache_dir(repo_id) / "snapshots" / revision)

    if validate_model_path(local_path):
        return local_path

    logger.warning(
        f"Cache symlinks appear broken for {repo_id}, "
        f"downloading to local directory without symlinks"
    )
    _clear_corrupted_cache(repo_id)
    _ensure_streams()

    try:
        local_dir = _get_local_model_dir(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id, local_dir=str(local_dir))
        local_path = str(local_dir)
    except Exception as e:
        raise ModelLoadError(
            f"Failed to download model files for {repo_id}: {e}"
        ) from e

    if not validate_model_path(local_path):
        raise ModelLoadError(
            f"Model files for {repo_id} could not be downloaded "
            f"successfully. Please manually delete the cache "
            f"directory and try again."
        )

    return local_path


def load_whisper_s2t_model(
    model_name: str,
    precision: str,
    device: str,
    beam_size: int = 1,
    local_path: str | None = None,
):
    """Invoke whisper_s2t.load_model with the right kwargs for the target model."""
    info = WHISPER_MODELS.get(ModelMetadata.resolve_model_key(model_name, precision))
    if info is None:
        raise ModelLoadError(
            f"Unknown model/precision combination: {model_name} - {precision}"
        )

    cpu_threads = get_optimal_cpu_threads() if device == "cpu" else 4

    kwargs = {
        "model_identifier": local_path or info["repo_id"],
        "device": device,
        "compute_type": precision,
        "asr_options": {"beam_size": max(1, int(beam_size))},
        "cpu_threads": cpu_threads,
    }

    if "large-v3" in info["repo_id"]:
        kwargs["n_mels"] = 128

    logger.info(
        f"Loading WhisperS2T model: name={model_name}, precision={precision}, "
        f"device={device}, beam_size={beam_size}"
    )

    try:
        model = whisper_s2t.load_model(**kwargs)
    except Exception as e:
        logger.exception(f"Failed to load WhisperS2T model {model_name}")
        raise ModelLoadError(f"Error loading model: {e}") from e

    logger.info(f"WhisperS2T model ready: {model_name} ({precision}) on {device}")
    return model
