import os
import gc
from PySide6.QtCore import QThread, Signal, QElapsedTimer
from pathlib import Path
import whisper_s2t
from queue import Queue
import torch
from threading import Event

class Worker(QThread):
    finished = Signal(str)
    progress = Signal(str)

    def __init__(self, directory, recursive, output_format, device, size, quantization, beam_size, batch_size, task, selected_extensions):
        super().__init__()
        self.directory = directory
        self.recursive = recursive
        self.output_format = output_format
        self.device = device
        self.size = size
        self.quantization = quantization
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.task = task.lower()
        self.selected_extensions = selected_extensions
        self.file_queue = Queue()
        self.enumeration_done = False
        self.stop_requested = Event()
        self.total_files = 0

    def request_stop(self):
        self.stop_requested.set()

    def release_transcriber_resources(self, model):
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def enqueue_files(self, directory_path, patterns):
        for pattern in patterns:
            if self.recursive:
                for path in directory_path.rglob(pattern):
                    self.file_queue.put(path)
                    self.total_files += 1
            else:
                for path in directory_path.glob(pattern):
                    self.file_queue.put(path)
                    self.total_files += 1
        self.enumeration_done = True

    def run(self):
        directory_path = Path(self.directory)
        patterns = [f'*{ext}' for ext in self.selected_extensions]

        self.enqueue_files(directory_path, patterns)

        model_identifier = f"ctranslate2-4you/whisper-{self.size}-ct2-{self.quantization}"
        model = whisper_s2t.load_model(model_identifier=model_identifier, backend='CTranslate2', device=self.device, compute_type=self.quantization, asr_options={'beam_size': self.beam_size}, cpu_threads=os.cpu_count())

        timer = QElapsedTimer()
        timer.start()

        processed_files = 0

        while not self.file_queue.empty() or not self.enumeration_done:
            if self.stop_requested.is_set():
                break
            try:
                audio_file = self.file_queue.get(timeout=1)
                processed_files += 1
                progress_message = f"Processing {audio_file} ({processed_files}/{self.total_files})"
                self.progress.emit(progress_message)
                out = model.transcribe_with_vad([str(audio_file)], lang_codes=['en'], tasks=[self.task], initial_prompts=[None], batch_size=self.batch_size)
                output_file_path = str(audio_file.with_suffix(f'.{self.output_format}'))
                whisper_s2t.write_outputs(out, format=self.output_format, op_files=[output_file_path])
                completion_message = f"Completed {audio_file} to {output_file_path} ({processed_files}/{self.total_files})"
                self.progress.emit(completion_message)
                self.file_queue.task_done()
            except Exception as e:
                error_message = f"Error processing file {audio_file if 'audio_file' in locals() else 'unknown'}: {e} ({processed_files}/{self.total_files})"
                self.progress.emit(error_message)
                print(f"\033[33m{error_message}\033[0m")
        
        self.release_transcriber_resources(model)

        processing_time = timer.elapsed() / 1000.0
        self.finished.emit(f"Total processing time: {processing_time:.2f} seconds")
