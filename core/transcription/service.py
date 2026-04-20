from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from core.logging_config import get_logger
from core.output.writers import SegmentData, TranscriptionResult
from core.temp_file_manager import temp_file_manager
from core.text.curation import curate_text

logger = get_logger(__name__)


def _to_segments(raw_segments: list[dict]) -> list[SegmentData]:
    segments: list[SegmentData] = []
    for s in raw_segments:
        if not isinstance(s, dict):
            continue
        text = s.get("text", "")
        start = float(s.get("start_time", 0.0) or 0.0)
        end = float(s.get("end_time", start) or start)
        segments.append(SegmentData(start=start, end=end, text=text))
    return segments


def _segments_to_text(segments: list[SegmentData]) -> str:
    return "\n".join(seg.text.lstrip() for seg in segments if seg.text)


class _TranscriberSignals(QObject):
    transcription_done = Signal(str)
    transcription_done_with_result = Signal(object)
    progress_updated = Signal(int, int, float)
    error_occurred = Signal(str)
    cancelled = Signal()


class _TranscriptionRunnable(QRunnable):
    def __init__(
        self,
        model,
        model_version: str,
        audio_file: str | Path,
        language: str = "en",
        task_mode: str = "transcribe",
        is_temp_file: bool = True,
        batch_size: int | None = None,
        get_current_version_func: Optional[Callable[[], str]] = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.model = model
        self.model_version = model_version
        self.audio_file = Path(audio_file)
        self.language = language or "en"
        self.task_mode = task_mode
        self.is_temp_file = is_temp_file
        self.batch_size = batch_size if batch_size and batch_size > 0 else 8
        self.get_current_version = get_current_version_func
        self.cancel_event = cancel_event or threading.Event()
        self.signals = _TranscriberSignals()

    def _is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def run(self) -> None:
        try:
            if self._is_cancelled():
                logger.info("Transcription cancelled before starting")
                self.signals.cancelled.emit()
                return

            if self.get_current_version and self.get_current_version() != self.model_version:
                logger.warning("Model changed during transcription setup")
                self.signals.cancelled.emit()
                return

            logger.info(
                f"Starting transcription: {self.audio_file.name} "
                f"(lang={self.language}, task={self.task_mode}, batch={self.batch_size})"
            )

            out = self.model.transcribe_with_vad(
                [str(self.audio_file)],
                lang_codes=[self.language],
                tasks=[self.task_mode],
                initial_prompts=[None],
                batch_size=self.batch_size,
            )

            if self._is_cancelled():
                self.signals.cancelled.emit()
                return

            raw_segments = out[0] if out else []
            segments = _to_segments(raw_segments)
            text = _segments_to_text(segments)

            duration = segments[-1].end if segments else None
            result = TranscriptionResult(
                text=text,
                segments=segments,
                language=self.language,
                duration=duration,
                source_file=self.audio_file,
            )

            self.signals.progress_updated.emit(len(segments), len(segments), 100.0)
            self.signals.transcription_done.emit(text)
            self.signals.transcription_done_with_result.emit(result)

        except Exception as e:
            if self._is_cancelled():
                logger.info("Transcription cancelled during processing")
                self.signals.cancelled.emit()
            else:
                logger.exception("Transcription failed")
                self.signals.error_occurred.emit(f"Transcription failed: {e}")
        finally:
            if self.is_temp_file:
                temp_file_manager.release(self.audio_file)


class TranscriptionService(QObject):
    transcription_started = Signal()
    transcription_completed = Signal(str)
    transcription_completed_with_result = Signal(object)
    transcription_progress = Signal(int, int, float)
    transcription_error = Signal(str)
    transcription_cancelled = Signal()

    def __init__(self, curate_text_enabled: bool = True, task_mode: str = "transcribe",
                 language: str = "en"):
        super().__init__()
        self.curate_enabled = curate_text_enabled
        self.task_mode = task_mode
        self.language = language
        self._thread_pool = QThreadPool.globalInstance()
        self._get_model_version_func: Optional[Callable[[], str]] = None
        self._cancel_event: threading.Event | None = None
        self._is_transcribing = False

    def set_model_version_provider(self, func: Callable[[], str]) -> None:
        self._get_model_version_func = func

    def set_task_mode(self, mode: str) -> None:
        self.task_mode = mode
        logger.debug(f"Task mode set to: {mode}")

    def set_language(self, lang: str) -> None:
        self.language = lang
        logger.debug(f"Language set to: {lang}")

    def set_curation_enabled(self, enabled: bool) -> None:
        self.curate_enabled = enabled

    def is_transcribing(self) -> bool:
        return self._is_transcribing

    def cancel_transcription(self) -> bool:
        if self._cancel_event and self._is_transcribing:
            logger.info("Cancellation requested")
            self._cancel_event.set()
            return True
        return False

    def transcribe_file(
        self,
        model,
        model_version: str,
        audio_file: str | Path,
        is_temp_file: bool = True,
        batch_size: int | None = None,
        language: str | None = None,
        task_mode: str | None = None,
    ) -> None:
        if not model:
            error_msg = "No model available for transcription"
            logger.error(error_msg)
            self.transcription_error.emit(error_msg)
            if is_temp_file:
                temp_file_manager.release(Path(audio_file))
            return

        try:
            self._cancel_event = threading.Event()
            self._is_transcribing = True

            runnable = _TranscriptionRunnable(
                model=model,
                model_version=model_version,
                audio_file=str(audio_file),
                language=language or self.language,
                task_mode=task_mode or self.task_mode,
                is_temp_file=is_temp_file,
                batch_size=batch_size,
                get_current_version_func=self._get_model_version_func,
                cancel_event=self._cancel_event,
            )
            runnable.signals.transcription_done.connect(self._on_transcription_done)
            runnable.signals.transcription_done_with_result.connect(
                self._on_transcription_done_with_result
            )
            runnable.signals.progress_updated.connect(self._on_progress_updated)
            runnable.signals.error_occurred.connect(self._on_transcription_error)
            runnable.signals.cancelled.connect(self._on_transcription_cancelled)
            self._thread_pool.start(runnable)
            self.transcription_started.emit()
        except Exception as e:
            logger.exception("Failed to start transcription")
            self._is_transcribing = False
            self.transcription_error.emit(f"Failed to start transcription: {e}")
            if is_temp_file:
                temp_file_manager.release(Path(audio_file))

    def _on_transcription_done(self, text: str) -> None:
        self._is_transcribing = False
        if self.curate_enabled:
            try:
                text = curate_text(text)
            except Exception as e:
                logger.warning(f"Text curation failed: {e}")

        self.transcription_completed.emit(text)

    def _on_transcription_done_with_result(self, result: object) -> None:
        self.transcription_completed_with_result.emit(result)

    def _on_progress_updated(self, segment_num: int, total_segments: int, percent: float) -> None:
        self.transcription_progress.emit(segment_num, total_segments, percent)

    def _on_transcription_error(self, error: str) -> None:
        self._is_transcribing = False
        logger.error(f"Transcription error: {error}")
        self.transcription_error.emit(error)

    def _on_transcription_cancelled(self) -> None:
        self._is_transcribing = False
        logger.info("Transcription was cancelled")
        self.transcription_cancelled.emit()

    def cleanup(self) -> None:
        import time as _time

        _t = _time.perf_counter()
        if self._cancel_event:
            self._cancel_event.set()
        logger.info(f"[SHUTDOWN]   TS cancel_event.set(): {_time.perf_counter() - _t:.3f}s")

        _t = _time.perf_counter()
        self._thread_pool.waitForDone(5000)
        logger.info(f"[SHUTDOWN]   TS waitForDone(5000): {_time.perf_counter() - _t:.3f}s")

        logger.debug("TranscriptionService cleanup complete")
