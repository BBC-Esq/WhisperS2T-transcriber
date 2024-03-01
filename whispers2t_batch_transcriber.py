import os
from PySide6.QtCore import QThread, Signal
from pathlib import Path
import whisper_s2t

class Worker(QThread):
    finished = Signal()
    progress = Signal(str)

    def __init__(self, directory, recursive, output_format, device, size, quantization, beam_size, batch_size):
        super().__init__()
        self.directory = directory
        self.recursive = recursive
        self.output_format = output_format
        self.device = device
        self.size = size
        self.quantization = quantization
        self.beam_size = beam_size
        self.batch_size = batch_size

    def run(self):
        directory_path = Path(self.directory)
        patterns = ['*.mp3', '*.wav', '*.flac', '*.wma']
        audio_files = []

        if self.recursive:
            for pattern in patterns:
                audio_files.extend(directory_path.rglob(pattern))
        else:
            for pattern in patterns:
                audio_files.extend(directory_path.glob(pattern))

        max_threads = os.cpu_count()
        cpu_threads = max(max_threads - 8, 2) if max_threads is not None else 2
        
        model_identifier = f"ctranslate2-4you/whisper-{self.size}-ct2-{self.quantization}"

        model = whisper_s2t.load_model(model_identifier=model_identifier, backend='CTranslate2', device=self.device, compute_type=self.quantization, asr_options={'beam_size': self.beam_size}, cpu_threads=cpu_threads)

        audio_files_str = [str(file) for file in audio_files]
        output_file_paths = [str(file.with_suffix(f'.{self.output_format}')) for file in audio_files]

        lang_codes = 'en'
        tasks = 'transcribe'
        initial_prompts = None

        if audio_files_str:
            self.progress.emit(f"Processing {len(audio_files_str)} files...")
            out = model.transcribe_with_vad(audio_files_str, lang_codes=lang_codes, tasks=tasks, initial_prompts=initial_prompts, batch_size=self.batch_size)
            whisper_s2t.write_outputs(out, format=self.output_format, op_files=output_file_paths)

            for original_audio_file, output_file_path in zip(audio_files, output_file_paths):
                self.progress.emit(f"Transcribed {original_audio_file} to {output_file_path}")

        self.finished.emit()
