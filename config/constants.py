from collections import OrderedDict

_MODEL_SPECS = [
    ("Whisper large-v3 turbo",  "whisper-large-v3-turbo",  "float32", 155, 4, "3.2 GB"),
    ("Whisper large-v3 turbo",  "whisper-large-v3-turbo",  "bfloat16", 160, 4, "3.0 GB"),
    ("Whisper large-v3 turbo",  "whisper-large-v3-turbo",  "float16", 165, 4, "2.8 GB"),
    ("Distil Whisper large-v3", "distil-whisper-large-v3", "float32", 160, 4, "3.0 GB"),
    ("Distil Whisper large-v3", "distil-whisper-large-v3", "bfloat16", 160, 4, "3.0 GB"),
    ("Distil Whisper large-v3", "distil-whisper-large-v3", "float16", 160, 4, "3.0 GB"),
    ("Whisper large-v3",        "whisper-large-v3",        "float32", 85,  2, "5.5 GB"),
    ("Whisper large-v3",        "whisper-large-v3",        "bfloat16", 95, 3, "3.8 GB"),
    ("Whisper large-v3",        "whisper-large-v3",        "float16", 100, 3, "3.3 GB"),
    ("Distil Whisper medium.en", "distil-whisper-medium.en", "float32", 160, 4, "3.0 GB"),
    ("Distil Whisper medium.en", "distil-whisper-medium.en", "bfloat16", 160, 4, "3.0 GB"),
    ("Distil Whisper medium.en", "distil-whisper-medium.en", "float16", 160, 4, "3.0 GB"),
    ("Whisper medium",          "whisper-medium",          "float32", 125, 5, "2.8 GB"),
    ("Whisper medium",          "whisper-medium",          "bfloat16", 135, 6, "2.2 GB"),
    ("Whisper medium",          "whisper-medium",          "float16", 140, 6, "2.0 GB"),
    ("Whisper medium.en",       "whisper-medium.en",       "float32", 130, 6, "2.5 GB"),
    ("Whisper medium.en",       "whisper-medium.en",       "bfloat16", 140, 7, "2.0 GB"),
    ("Whisper medium.en",       "whisper-medium.en",       "float16", 145, 7, "1.8 GB"),
    ("Distil Whisper small.en", "distil-whisper-small.en", "float32", 160, 4, "3.0 GB"),
    ("Distil Whisper small.en", "distil-whisper-small.en", "bfloat16", 160, 4, "3.0 GB"),
    ("Distil Whisper small.en", "distil-whisper-small.en", "float16", 160, 4, "3.0 GB"),
    ("Whisper small",           "whisper-small",           "float32", 175, 12, "1.8 GB"),
    ("Whisper small",           "whisper-small",           "bfloat16", 185, 13, "1.4 GB"),
    ("Whisper small",           "whisper-small",           "float16", 190, 13, "1.3 GB"),
    ("Whisper small.en",        "whisper-small.en",        "float32", 180, 14, "1.5 GB"),
    ("Whisper small.en",        "whisper-small.en",        "bfloat16", 190, 15, "1.2 GB"),
    ("Whisper small.en",        "whisper-small.en",        "float16", 195, 15, "1.1 GB"),
    ("Whisper base",            "whisper-base",            "float32", 225, 20, "1.1 GB"),
    ("Whisper base",            "whisper-base",            "bfloat16", 235, 21, "0.9 GB"),
    ("Whisper base",            "whisper-base",            "float16", 240, 21, "0.85 GB"),
    ("Whisper base.en",         "whisper-base.en",         "float32", 230, 22, "1.0 GB"),
    ("Whisper base.en",         "whisper-base.en",         "bfloat16", 240, 23, "0.85 GB"),
    ("Whisper base.en",         "whisper-base.en",         "float16", 245, 23, "0.8 GB"),
    ("Whisper tiny",            "whisper-tiny",            "float32", 275, 28, "0.75 GB"),
    ("Whisper tiny",            "whisper-tiny",            "bfloat16", 285, 29, "0.65 GB"),
    ("Whisper tiny",            "whisper-tiny",            "float16", 290, 29, "0.6 GB"),
    ("Whisper tiny.en",         "whisper-tiny.en",         "float32", 280, 30, "0.7 GB"),
    ("Whisper tiny.en",         "whisper-tiny.en",         "bfloat16", 290, 31, "0.6 GB"),
    ("Whisper tiny.en",         "whisper-tiny.en",         "float16", 295, 31, "0.55 GB"),
]

WHISPER_MODELS = {
    f"{name} - {prec}": {
        'name': name,
        'precision': prec,
        'repo_id': f'ctranslate2-4you/{slug}-ct2-{prec}',
        'tokens_per_second': tps,
        'optimal_batch_size': batch,
        'avg_vram_usage': vram,
    }
    for name, slug, prec, tps, batch, vram in _MODEL_SPECS
}

MODEL_NAMES = list(OrderedDict.fromkeys(name for name, *_ in _MODEL_SPECS))

MODEL_PRECISIONS = {}
for name, slug, prec, *_ in _MODEL_SPECS:
    MODEL_PRECISIONS.setdefault(name, []).append(prec)

DISTIL_MODELS = frozenset(name for name, *_ in _MODEL_SPECS if name.startswith("Distil"))

WHISPER_LANGUAGES = OrderedDict([
    ("af", "Afrikaans"), ("am", "Amharic"), ("ar", "Arabic"), ("as", "Assamese"),
    ("az", "Azerbaijani"), ("ba", "Bashkir"), ("be", "Belarusian"), ("bg", "Bulgarian"),
    ("bn", "Bengali"), ("bo", "Tibetan"), ("br", "Breton"), ("bs", "Bosnian"),
    ("ca", "Catalan"), ("cs", "Czech"), ("cy", "Welsh"), ("da", "Danish"),
    ("de", "German"), ("el", "Greek"), ("en", "English"), ("es", "Spanish"),
    ("et", "Estonian"), ("eu", "Basque"), ("fa", "Persian"), ("fi", "Finnish"),
    ("fo", "Faroese"), ("fr", "French"), ("gl", "Galician"), ("gu", "Gujarati"),
    ("ha", "Hausa"), ("haw", "Hawaiian"), ("he", "Hebrew"), ("hi", "Hindi"),
    ("hr", "Croatian"), ("ht", "Haitian Creole"), ("hu", "Hungarian"), ("hy", "Armenian"),
    ("id", "Indonesian"), ("is", "Icelandic"), ("it", "Italian"), ("ja", "Japanese"),
    ("jw", "Javanese"), ("ka", "Georgian"), ("kk", "Kazakh"), ("km", "Khmer"),
    ("kn", "Kannada"), ("ko", "Korean"), ("la", "Latin"), ("lb", "Luxembourgish"),
    ("ln", "Lingala"), ("lo", "Lao"), ("lt", "Lithuanian"), ("lv", "Latvian"),
    ("mg", "Malagasy"), ("mi", "Maori"), ("mk", "Macedonian"), ("ml", "Malayalam"),
    ("mn", "Mongolian"), ("mr", "Marathi"), ("ms", "Malay"), ("mt", "Maltese"),
    ("my", "Myanmar"), ("ne", "Nepali"), ("nl", "Dutch"), ("nn", "Nynorsk"),
    ("no", "Norwegian"), ("oc", "Occitan"), ("pa", "Punjabi"), ("pl", "Polish"),
    ("ps", "Pashto"), ("pt", "Portuguese"), ("ro", "Romanian"), ("ru", "Russian"),
    ("sa", "Sanskrit"), ("sd", "Sindhi"), ("si", "Sinhala"), ("sk", "Slovak"),
    ("sl", "Slovenian"), ("sn", "Shona"), ("so", "Somali"), ("sq", "Albanian"),
    ("sr", "Serbian"), ("su", "Sundanese"), ("sv", "Swedish"), ("sw", "Swahili"),
    ("ta", "Tamil"), ("te", "Telugu"), ("tg", "Tajik"), ("th", "Thai"),
    ("tk", "Turkmen"), ("tl", "Tagalog"), ("tr", "Turkish"), ("tt", "Tatar"),
    ("uk", "Ukrainian"), ("ur", "Urdu"), ("uz", "Uzbek"), ("vi", "Vietnamese"),
    ("yi", "Yiddish"), ("yo", "Yoruba"), ("zh", "Chinese"),
])

SUPPORTED_AUDIO_EXTENSIONS = [
    ".aac", ".amr", ".asf", ".avi", ".flac", ".m4a",
    ".mkv", ".mp3", ".mp4", ".wav", ".webm", ".wma"
]

OUTPUT_FORMATS = ["txt", "vtt", "srt", "tsv", "json"]
TASK_MODES = ["transcribe", "translate"]

DEFAULT_BEAM_SIZE = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_OUTPUT_FORMAT = "txt"
DEFAULT_TASK_MODE = "transcribe"
DEFAULT_LANGUAGE = "en"
