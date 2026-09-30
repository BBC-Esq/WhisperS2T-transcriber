from collections import OrderedDict

_MODEL_SPECS = [
    ("Whisper large-v3 turbo",  "whisper-large-v3-turbo",  "float32"),
    ("Whisper large-v3 turbo",  "whisper-large-v3-turbo",  "bfloat16"),
    ("Whisper large-v3 turbo",  "whisper-large-v3-turbo",  "float16"),
    ("Distil Whisper large-v3.5", "whisper-distil-large-v3.5", "float32"),
    ("Distil Whisper large-v3.5", "whisper-distil-large-v3.5", "bfloat16"),
    ("Distil Whisper large-v3.5", "whisper-distil-large-v3.5", "float16"),
    ("Distil Whisper large-v3", "distil-whisper-large-v3", "float32"),
    ("Distil Whisper large-v3", "distil-whisper-large-v3", "bfloat16"),
    ("Distil Whisper large-v3", "distil-whisper-large-v3", "float16"),
    ("Whisper large-v3",        "whisper-large-v3",        "float32"),
    ("Whisper large-v3",        "whisper-large-v3",        "bfloat16"),
    ("Whisper large-v3",        "whisper-large-v3",        "float16"),
    ("Distil Whisper medium.en", "distil-whisper-medium.en", "float32"),
    ("Distil Whisper medium.en", "distil-whisper-medium.en", "bfloat16"),
    ("Distil Whisper medium.en", "distil-whisper-medium.en", "float16"),
    ("Whisper medium",          "whisper-medium",          "float32"),
    ("Whisper medium",          "whisper-medium",          "bfloat16"),
    ("Whisper medium",          "whisper-medium",          "float16"),
    ("Whisper medium.en",       "whisper-medium.en",       "float32"),
    ("Whisper medium.en",       "whisper-medium.en",       "bfloat16"),
    ("Whisper medium.en",       "whisper-medium.en",       "float16"),
    ("Distil Whisper small.en", "distil-whisper-small.en", "float32"),
    ("Distil Whisper small.en", "distil-whisper-small.en", "bfloat16"),
    ("Distil Whisper small.en", "distil-whisper-small.en", "float16"),
    ("Whisper small",           "whisper-small",           "float32"),
    ("Whisper small",           "whisper-small",           "bfloat16"),
    ("Whisper small",           "whisper-small",           "float16"),
    ("Whisper small.en",        "whisper-small.en",        "float32"),
    ("Whisper small.en",        "whisper-small.en",        "bfloat16"),
    ("Whisper small.en",        "whisper-small.en",        "float16"),
    ("Whisper base",            "whisper-base",            "float32"),
    ("Whisper base",            "whisper-base",            "bfloat16"),
    ("Whisper base",            "whisper-base",            "float16"),
    ("Whisper base.en",         "whisper-base.en",         "float32"),
    ("Whisper base.en",         "whisper-base.en",         "bfloat16"),
    ("Whisper base.en",         "whisper-base.en",         "float16"),
    ("Whisper tiny",            "whisper-tiny",            "float32"),
    ("Whisper tiny",            "whisper-tiny",            "bfloat16"),
    ("Whisper tiny",            "whisper-tiny",            "float16"),
    ("Whisper tiny.en",         "whisper-tiny.en",         "float32"),
    ("Whisper tiny.en",         "whisper-tiny.en",         "bfloat16"),
    ("Whisper tiny.en",         "whisper-tiny.en",         "float16"),
]

WHISPER_MODELS = {
    f"{name} - {prec}": {
        'name': name,
        'precision': prec,
        'repo_id': f'ctranslate2-4you/{slug}-ct2-{prec}',
    }
    for name, slug, prec in _MODEL_SPECS
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
    ".mkv", ".mp3", ".mp4", ".ogg", ".wav", ".webm", ".wma",
]

OUTPUT_FORMATS = ["txt", "vtt", "srt", "json"]
TASK_MODES = ["transcribe", "translate"]

DEFAULT_BEAM_SIZE = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_OUTPUT_FORMAT = "txt"
DEFAULT_TASK_MODE = "transcribe"
DEFAULT_LANGUAGE = "en"
