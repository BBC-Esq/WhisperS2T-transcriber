from pathlib import Path
import whisper_s2t

def transcribe_audio_files(directory):
    model_kwargs = {
        'compute_type': 'float16',
        'asr_options': {
            "beam_size": 5,
            "best_of": 1,
            "patience": 2,
            "length_penalty": 1,
            "repetition_penalty": 1.01,
            "no_repeat_ngram_size": 0,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.5,
            "prefix": None,
            "suppress_blank": True,
            "suppress_tokens": [-1],
            "without_timestamps": False,
            "max_initial_timestamp": 1.0,
            "word_timestamps": True,
            "sampling_temperature": 1.0,
            "return_scores": True,
            "return_no_speech_prob": True,
            "word_aligner_model": 'tiny',
        },
        "device": "cuda",
    }

    model = whisper_s2t.load_model(model_identifier="medium.en", backend='CTranslate2', **model_kwargs)

    directory_path = Path(directory)
    patterns = ['*.mp3', '*.wav', '*.flac']
    audio_files = []

    # Use rglob for recursive globbing
    for pattern in patterns:
        audio_files.extend(directory_path.rglob(pattern))

    # Convert audio file paths to strings and prepare VTT file paths
    audio_files_str = [str(file) for file in audio_files]
    vtt_file_paths = [str(file.with_suffix('.vtt')) for file in audio_files]

    lang_codes = 'en'
    tasks = 'transcribe'
    initial_prompts = None

    out = model.transcribe_with_vad(audio_files_str, lang_codes=lang_codes, tasks=tasks, initial_prompts=initial_prompts, batch_size=70)
    whisper_s2t.write_outputs(out, format='vtt', op_files=vtt_file_paths)

    for original_audio_file, vtt_file_path in zip(audio_files, vtt_file_paths):
        print(f"Transcribed {original_audio_file} to {vtt_file_path}")

audio_files_directory = r"C:\PATH\Scripts\WhisperS2T-batch-process\test"
transcribe_audio_files(audio_files_directory)
