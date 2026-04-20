from __future__ import annotations

from pathlib import Path
from threading import Event

from PySide6.QtCore import QElapsedTimer, QThread, Signal

from core.logging_config import get_logger
from core.output.writers import SegmentData, TranscriptionResult, write_output

logger = get_logger(__name__)


def _is_oom_error(exc: Exception) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg or ("cuda" in msg and "alloc" in msg):
            return True
    return False


def _deduplicated_output_path(output_dir: Path, stem: str, suffix: str,
                              seen: dict[str, int]) -> Path:
    key = stem.lower()
    if key in seen:
        seen[key] += 1
        return output_dir / f"{stem}_{seen[key]}{suffix}"
    else:
        seen[key] = 0
        return output_dir / f"{stem}{suffix}"


def _segments_from_whisper_s2t(raw_segments: list) -> list[SegmentData]:
    segments: list[SegmentData] = []
    for s in raw_segments:
        if not isinstance(s, dict):
            continue
        text = s.get("text", "")
        start = float(s.get("start_time", 0.0) or 0.0)
        end = float(s.get("end_time", start) or start)
        segments.append(SegmentData(start=start, end=end, text=text))
    return segments


class BatchProcessor(QThread):

    progress = Signal(int, int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        files: list[Path],
        model,
        output_format: str,
        output_directory: str | None,
        batch_size: int,
        language: str,
        task_mode: str,
    ):
        super().__init__()
        self.files = [Path(f) for f in files]
        self.model = model
        self.output_format = output_format
        self.output_directory = output_directory
        self.batch_size = batch_size if batch_size and batch_size > 0 else 8
        self.language = language or "en"
        self.task_mode = task_mode or "transcribe"
        self.stop_requested = Event()

    def request_stop(self) -> None:
        self.stop_requested.set()

    def run(self) -> None:
        timer = QElapsedTimer()
        timer.start()

        seen_names: dict[str, int] = {}

        try:
            total_files = len(self.files)

            for idx, audio_file in enumerate(self.files, 1):
                if self.stop_requested.is_set():
                    break

                self.progress.emit(idx, total_files, f"Processing {audio_file.name}")

                try:
                    out = self.model.transcribe_with_vad(
                        [str(audio_file)],
                        lang_codes=[self.language],
                        tasks=[self.task_mode],
                        initial_prompts=[None],
                        batch_size=self.batch_size,
                    )

                    if self.stop_requested.is_set():
                        break

                    raw_segments = out[0] if out else []
                    segments = _segments_from_whisper_s2t(raw_segments)
                    text = "\n".join(seg.text.lstrip() for seg in segments if seg.text)

                    duration = segments[-1].end if segments else None
                    result = TranscriptionResult(
                        text=text,
                        segments=segments,
                        language=self.language,
                        duration=duration,
                        source_file=audio_file,
                    )

                    out_suffix = f".{self.output_format}"
                    if self.output_directory:
                        out_dir = Path(self.output_directory)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        output_file = _deduplicated_output_path(
                            out_dir, audio_file.stem, out_suffix, seen_names
                        )
                    else:
                        output_file = audio_file.with_suffix(out_suffix)

                    write_output(result, output_file, self.output_format)

                    self.progress.emit(
                        idx, total_files, f"Completed {audio_file.name}"
                    )

                except Exception as e:
                    if _is_oom_error(e):
                        self.error.emit(
                            f"GPU out of memory processing {audio_file.name}: {e}\n"
                            "Stopping batch. Try a smaller model or reduce batch size."
                        )
                        logger.error("OOM error, stopping batch: %s", e)
                        break
                    self.error.emit(f"Error processing {audio_file.name}: {e}")
                    logger.error("Error processing %s: %s", audio_file.name, e)

        except Exception as e:
            self.error.emit(f"Processing failed: {e}")
            logger.exception("Batch processing failed")

        finally:
            elapsed = timer.elapsed() / 1000.0
            self.finished.emit(f"Processing time: {elapsed:.2f} seconds")
