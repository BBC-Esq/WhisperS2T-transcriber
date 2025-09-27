"""Model definitions and application constants."""

WHISPER_MODELS = {
    # LARGE-V3
    'Distil Whisper large-v3 - float32': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-float32',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper large-v3 - bfloat16': {
        'name': 'Distil Whisper large-v3',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-bfloat16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper large-v3 - float16': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-float16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Whisper large-v3 - float32': {
        'name': 'Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-float32',
        'tokens_per_second': 85,
        'optimal_batch_size': 2,
        'avg_vram_usage': '5.5 GB'
    },
    'Whisper large-v3 - bfloat16': {
        'name': 'Whisper large-v3',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-bfloat16',
        'tokens_per_second': 95,
        'optimal_batch_size': 3,
        'avg_vram_usage': '3.8 GB'
    },
    'Whisper large-v3 - float16': {
        'name': 'Whisper large-v3',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-float16',
        'tokens_per_second': 100,
        'optimal_batch_size': 3,
        'avg_vram_usage': '3.3 GB'
    },
    # MEDIUM.EN
    'Distil Whisper medium.en - float32': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-float32',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper medium.en - bfloat16': {
        'name': 'Distil Whisper medium.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-bfloat16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper medium.en - float16': {
        'name': 'Distil Whisper medium.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-float16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Whisper medium.en - float32': {
        'name': 'Whisper medium.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-float32',
        'tokens_per_second': 130,
        'optimal_batch_size': 6,
        'avg_vram_usage': '2.5 GB'
    },
    'Whisper medium.en - bfloat16': {
        'name': 'Whisper medium.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-bfloat16',
        'tokens_per_second': 140,
        'optimal_batch_size': 7,
        'avg_vram_usage': '2.0 GB'
    },
    'Whisper medium.en - float16': {
        'name': 'Whisper medium.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-float16',
        'tokens_per_second': 145,
        'optimal_batch_size': 7,
        'avg_vram_usage': '1.8 GB'
    },
    # SMALL.EN
    'Distil Whisper small.en - float32': {
        'name': 'Distil Whisper small.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-float32',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper small.en - bfloat16': {
        'name': 'Distil Whisper small.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-bfloat16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Distil Whisper small.en - float16': {
        'name': 'Distil Whisper small.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-float16',
        'tokens_per_second': 160,
        'optimal_batch_size': 4,
        'avg_vram_usage': '3.0 GB'
    },
    'Whisper small.en - float32': {
        'name': 'Whisper small.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-float32',
        'tokens_per_second': 180,
        'optimal_batch_size': 14,
        'avg_vram_usage': '1.5 GB'
    },
    'Whisper small.en - bfloat16': {
        'name': 'Whisper small.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-bfloat16',
        'tokens_per_second': 190,
        'optimal_batch_size': 15,
        'avg_vram_usage': '1.2 GB'
    },
    'Whisper small.en - float16': {
        'name': 'Whisper small.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-float16',
        'tokens_per_second': 195,
        'optimal_batch_size': 15,
        'avg_vram_usage': '1.1 GB'
    },
    # BASE.EN
    'Whisper base.en - float32': {
        'name': 'Whisper base.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-float32',
        'tokens_per_second': 230,
        'optimal_batch_size': 22,
        'avg_vram_usage': '1.0 GB'
    },
    'Whisper base.en - bfloat16': {
        'name': 'Whisper base.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-bfloat16',
        'tokens_per_second': 240,
        'optimal_batch_size': 23,
        'avg_vram_usage': '0.85 GB'
    },
    'Whisper base.en - float16': {
        'name': 'Whisper base.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-float16',
        'tokens_per_second': 245,
        'optimal_batch_size': 23,
        'avg_vram_usage': '0.8 GB'
    },
    # TINY.EN
    'Whisper tiny.en - float32': {
        'name': 'Whisper tiny.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-float32',
        'tokens_per_second': 280,
        'optimal_batch_size': 30,
        'avg_vram_usage': '0.7 GB'
    },
    'Whisper tiny.en - bfloat16': {
        'name': 'Whisper tiny.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-bfloat16',
        'tokens_per_second': 290,
        'optimal_batch_size': 31,
        'avg_vram_usage': '0.6 GB'
    },
    'Whisper tiny.en - float16': {
        'name': 'Whisper tiny.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-float16',
        'tokens_per_second': 295,
        'optimal_batch_size': 31,
        'avg_vram_usage': '0.55 GB'
    },
}


# File extensions
SUPPORTED_AUDIO_EXTENSIONS = [
    ".aac", ".amr", ".asf", ".avi", ".flac", ".m4a", 
    ".mkv", ".mp3", ".mp4", ".wav", ".webm", ".wma"
]

# Output formats
OUTPUT_FORMATS = ["txt", "vtt", "srt", "tsv", "json"]

# Task modes
TASK_MODES = ["transcribe", "translate"]

# Default settings
DEFAULT_BEAM_SIZE = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_OUTPUT_FORMAT = "txt"
DEFAULT_TASK_MODE = "transcribe"