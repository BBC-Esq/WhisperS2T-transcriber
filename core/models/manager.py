from __future__ import annotations

import gc
import threading
import uuid
from typing import Optional

from PySide6.QtCore import QMutex, QMutexLocker, QObject, QRunnable, QThreadPool, Signal

from core.exceptions import ModelLoadError
from core.logging_config import get_logger
from core.models.loader import (
    check_model_cached,
    download_model_files,
    get_missing_files,
    get_repo_file_info,
    get_repo_id,
    load_whisper_s2t_model,
    validate_model_path,
)

logger = get_logger(__name__)


_NETWORK_ERROR_TERMS = [
    "connection",
    "network",
    "resolve",
    "urlerror",
    "timeout",
    "unreachable",
    "dns",
    "socket",
    "offline",
]


def _is_network_error(exception: Exception) -> bool:
    msg = str(exception).lower()
    return any(term in msg for term in _NETWORK_ERROR_TERMS)


def _release_model(model) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    del model
    gc.collect()


class _LoaderSignals(QObject):
    model_loaded = Signal(object, str, str, str, str)
    error_occurred = Signal(str, str)
    download_started = Signal(str, object, str)
    download_progress = Signal(object, object, str)
    download_finished = Signal(str, str)
    download_cancelled = Signal(str)
    loading_started = Signal(str, str)


class _ModelLoaderRunnable(QRunnable):
    def __init__(
        self,
        model_name: str,
        precision: str,
        device: str,
        beam_size: int,
        model_version: str,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.model_name = model_name
        self.precision = precision
        self.device = device
        self.beam_size = beam_size
        self.model_version = model_version
        self.cancel_event = cancel_event
        self.signals = _LoaderSignals()

    def run(self) -> None:
        try:
            repo_id = get_repo_id(self.model_name, self.precision)

            if self.cancel_event.is_set():
                self.signals.download_cancelled.emit(self.model_version)
                return

            local_path = self._resolve_model_files(repo_id)
            if local_path is None:
                return

            if self.cancel_event.is_set():
                self.signals.download_cancelled.emit(self.model_version)
                return

            self.signals.loading_started.emit(self.model_name, self.model_version)

            model = load_whisper_s2t_model(
                self.model_name,
                self.precision,
                self.device,
                beam_size=self.beam_size,
                local_path=local_path,
            )
            self.signals.model_loaded.emit(
                model,
                self.model_name,
                self.precision,
                self.device,
                self.model_version,
            )
        except ModelLoadError as e:
            logger.error(f"Model load error: {e}")
            self.signals.error_occurred.emit(str(e), self.model_version)
        except Exception as e:
            if self.cancel_event.is_set():
                self.signals.download_cancelled.emit(self.model_version)
            else:
                logger.exception("Unexpected error loading model")
                self.signals.error_occurred.emit(
                    f"Unexpected error: {e}", self.model_version
                )

    def _resolve_model_files(self, repo_id: str) -> Optional[str]:
        cached_path = check_model_cached(repo_id)

        if cached_path:
            files_info = None
            try:
                files_info = get_repo_file_info(repo_id)
            except Exception as e:
                if _is_network_error(e):
                    if validate_model_path(cached_path):
                        logger.info(
                            f"Offline but found cached model for "
                            f"'{self.model_name}', using cache as-is"
                        )
                        return cached_path
                    self.signals.error_occurred.emit(
                        f"Cached model '{self.model_name}' appears corrupted "
                        f"and cannot be verified offline. Please connect to "
                        f"the internet to re-download, or delete the cached "
                        f"model and try again.",
                        self.model_version,
                    )
                    return None
                self.signals.error_occurred.emit(
                    f"Failed to get model info for '{self.model_name}': {e}",
                    self.model_version,
                )
                return None

            if self.cancel_event.is_set():
                self.signals.download_cancelled.emit(self.model_version)
                return None

            _, missing_files = get_missing_files(repo_id, files_info, cached_path)

            if not missing_files:
                return cached_path

            return self._download_files(repo_id, missing_files)

        try:
            files_info = get_repo_file_info(repo_id)
        except Exception as e:
            if _is_network_error(e):
                self.signals.error_occurred.emit(
                    f"Cannot download model '{self.model_name}': "
                    f"No internet connection. Please connect to the "
                    f"internet or select a previously downloaded model.",
                    self.model_version,
                )
            else:
                self.signals.error_occurred.emit(
                    f"Failed to get model info for '{self.model_name}': {e}",
                    self.model_version,
                )
            return None

        if self.cancel_event.is_set():
            self.signals.download_cancelled.emit(self.model_version)
            return None

        _, missing_files = get_missing_files(repo_id, files_info, cached_path)

        if not missing_files:
            return check_model_cached(repo_id)

        return self._download_files(repo_id, missing_files)

    def _download_files(
        self, repo_id: str, files_to_download: list[tuple[str, int]]
    ) -> Optional[str]:
        total_bytes = sum(size for _, size in files_to_download)
        self.signals.download_started.emit(
            self.model_name, total_bytes, self.model_version
        )

        try:
            local_path = download_model_files(
                repo_id,
                files_to_download,
                progress_callback=self._on_download_progress,
                cancel_event=self.cancel_event,
            )
        except InterruptedError:
            self.signals.download_cancelled.emit(self.model_version)
            return None
        except Exception as e:
            if _is_network_error(e):
                self.signals.error_occurred.emit(
                    f"Download failed for '{self.model_name}': "
                    f"Network connection lost. Please check your "
                    f"internet connection and try again.",
                    self.model_version,
                )
            else:
                self.signals.error_occurred.emit(
                    f"Download failed for '{self.model_name}': {e}",
                    self.model_version,
                )
            return None

        self.signals.download_finished.emit(self.model_name, self.model_version)
        return local_path

    def _on_download_progress(self, downloaded: int, total: int) -> None:
        if not self.cancel_event.is_set():
            self.signals.download_progress.emit(downloaded, total, self.model_version)


class ModelManager(QObject):
    model_loaded = Signal(str, str, str)
    model_error = Signal(str)
    download_started = Signal(str, object)
    download_progress = Signal(object, object)
    download_finished = Signal(str)
    download_cancelled = Signal()
    loading_started = Signal(str)

    def __init__(self):
        super().__init__()
        self._model = None
        self._model_version: Optional[str] = None
        self._pending_version: Optional[str] = None
        self._model_mutex = QMutex()
        self._thread_pool = QThreadPool.globalInstance()
        self._current_settings: dict = {}
        self._cancel_event: Optional[threading.Event] = None

    def load_model(
        self, model_name: str, precision: str, device: str, beam_size: int = 1
    ) -> None:
        logger.info(
            f"Requesting model load: {model_name}, {precision}, {device}, beam={beam_size}"
        )

        if self._cancel_event:
            self._cancel_event.set()

        new_version = str(uuid.uuid4())
        self._pending_version = new_version
        self._cancel_event = threading.Event()

        runnable = _ModelLoaderRunnable(
            model_name, precision, device, beam_size, new_version, self._cancel_event
        )
        runnable.signals.model_loaded.connect(self._on_model_loaded)
        runnable.signals.error_occurred.connect(self._on_model_error)
        runnable.signals.download_started.connect(self._on_download_started)
        runnable.signals.download_progress.connect(self._on_download_progress)
        runnable.signals.download_finished.connect(self._on_download_finished)
        runnable.signals.download_cancelled.connect(self._on_download_cancelled)
        runnable.signals.loading_started.connect(self._on_loading_started)
        self._thread_pool.start(runnable)

    def cancel_loading(self) -> None:
        if self._cancel_event:
            self._cancel_event.set()

    def get_model(self) -> tuple[Optional[object], Optional[str]]:
        with QMutexLocker(self._model_mutex):
            return self._model, self._model_version

    def get_or_load_model_sync(
        self, model_name: str, precision: str, device: str, beam_size: int = 1
    ):
        """Synchronous variant used by the server API when it needs the current
        model right now (blocks until a load finishes; typically the model is
        already resident in memory)."""
        model, _ = self.get_model()
        if model is not None and self._current_settings == {
            "model_name": model_name,
            "precision": precision,
            "device_type": device,
        }:
            return model

        version = str(uuid.uuid4())
        cancel_event = threading.Event()
        runnable = _ModelLoaderRunnable(
            model_name, precision, device, beam_size, version, cancel_event
        )
        holder: dict = {}
        done_event = threading.Event()

        def _on_loaded(m, *_args):
            holder["model"] = m
            done_event.set()

        def _on_error(err, _v):
            holder["error"] = err
            done_event.set()

        def _on_cancelled(_v):
            holder["error"] = "cancelled"
            done_event.set()

        runnable.signals.model_loaded.connect(_on_loaded)
        runnable.signals.error_occurred.connect(_on_error)
        runnable.signals.download_cancelled.connect(_on_cancelled)
        self._thread_pool.start(runnable)

        done_event.wait()
        if "error" in holder:
            raise ModelLoadError(holder["error"])
        return holder.get("model")

    def _on_download_started(
        self, model_name: str, total_bytes: int, version: str
    ) -> None:
        if version == self._pending_version:
            self.download_started.emit(model_name, total_bytes)

    def _on_download_progress(
        self, downloaded: int, total: int, version: str
    ) -> None:
        if version == self._pending_version:
            self.download_progress.emit(downloaded, total)

    def _on_download_finished(self, model_name: str, version: str) -> None:
        if version == self._pending_version:
            self.download_finished.emit(model_name)

    def _on_download_cancelled(self, version: str) -> None:
        if version == self._pending_version:
            self.download_cancelled.emit()

    def _on_loading_started(self, model_name: str, version: str) -> None:
        if version == self._pending_version:
            self.loading_started.emit(model_name)

    def _on_model_loaded(
        self, model, name: str, precision: str, device: str, version: str
    ) -> None:
        if version != self._pending_version:
            logger.info(f"Ignoring stale model load (version {version})")
            _release_model(model)
            return

        with QMutexLocker(self._model_mutex):
            if self._model is not None:
                _release_model(self._model)
                self._model = None
            self._model = model
            self._model_version = version

        self._current_settings = {
            "model_name": name,
            "precision": precision,
            "device_type": device,
        }
        logger.info(f"Model loaded successfully: {name}")
        self.model_loaded.emit(name, precision, device)

    def _on_model_error(self, error: str, version: str) -> None:
        if version == self._pending_version:
            logger.error(f"Model error: {error}")
            self.model_error.emit(error)

    def cleanup(self) -> None:
        import time as _time

        _t = _time.perf_counter()
        if self._cancel_event:
            self._cancel_event.set()
        logger.info(f"[SHUTDOWN]   MM cancel_event.set(): {_time.perf_counter() - _t:.3f}s")

        _t = _time.perf_counter()
        self._thread_pool.waitForDone(5000)
        logger.info(f"[SHUTDOWN]   MM waitForDone(5000): {_time.perf_counter() - _t:.3f}s")

        _t = _time.perf_counter()
        with QMutexLocker(self._model_mutex):
            if self._model is not None:
                _release_model(self._model)
                self._model = None
                self._model_version = None
        logger.info(f"[SHUTDOWN]   MM unload+gc: {_time.perf_counter() - _t:.3f}s")

        logger.debug("ModelManager cleanup complete")
