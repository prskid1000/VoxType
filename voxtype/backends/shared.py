"""Cross-backend constants. Pulled out so multiple Whisper-family
backends (transformers, faster-whisper) can share the same 99-language
tokenizer table without duplication."""
from __future__ import annotations


WHISPER_LANGUAGES: list[tuple[str, str]] = [
    ("auto", "Auto-detect"),
    ("af",  "Afrikaans"), ("am",  "Amharic"),   ("ar",  "Arabic"),
    ("as",  "Assamese"),  ("az",  "Azerbaijani"),("ba",  "Bashkir"),
    ("be",  "Belarusian"),("bn",  "Bengali"),   ("bo",  "Tibetan"),
    ("br",  "Breton"),    ("bs",  "Bosnian"),   ("bg",  "Bulgarian"),
    ("ca",  "Catalan"),   ("cs",  "Czech"),     ("cy",  "Welsh"),
    ("da",  "Danish"),    ("de",  "German"),    ("el",  "Greek"),
    ("en",  "English"),   ("es",  "Spanish"),   ("et",  "Estonian"),
    ("eu",  "Basque"),    ("fa",  "Persian"),   ("fi",  "Finnish"),
    ("fo",  "Faroese"),   ("fr",  "French"),    ("gl",  "Galician"),
    ("gu",  "Gujarati"),  ("ha",  "Hausa"),     ("haw", "Hawaiian"),
    ("he",  "Hebrew"),    ("hi",  "Hindi"),     ("hr",  "Croatian"),
    ("ht",  "Haitian Creole"),
    ("hu",  "Hungarian"), ("hy",  "Armenian"),  ("id",  "Indonesian"),
    ("is",  "Icelandic"), ("it",  "Italian"),   ("ja",  "Japanese"),
    ("jw",  "Javanese"),  ("ka",  "Georgian"),  ("kk",  "Kazakh"),
    ("km",  "Khmer"),     ("kn",  "Kannada"),   ("ko",  "Korean"),
    ("la",  "Latin"),     ("lb",  "Luxembourgish"),
    ("ln",  "Lingala"),   ("lo",  "Lao"),       ("lt",  "Lithuanian"),
    ("lv",  "Latvian"),   ("mg",  "Malagasy"),  ("mi",  "Maori"),
    ("mk",  "Macedonian"),("ml",  "Malayalam"), ("mn",  "Mongolian"),
    ("mr",  "Marathi"),   ("ms",  "Malay"),     ("mt",  "Maltese"),
    ("my",  "Burmese"),   ("ne",  "Nepali"),    ("nl",  "Dutch"),
    ("nn",  "Norwegian Nynorsk"),
    ("no",  "Norwegian"), ("oc",  "Occitan"),   ("pa",  "Punjabi"),
    ("pl",  "Polish"),    ("ps",  "Pashto"),    ("pt",  "Portuguese"),
    ("ro",  "Romanian"),  ("ru",  "Russian"),   ("sa",  "Sanskrit"),
    ("sd",  "Sindhi"),    ("si",  "Sinhala"),   ("sk",  "Slovak"),
    ("sl",  "Slovenian"), ("sn",  "Shona"),     ("so",  "Somali"),
    ("sq",  "Albanian"),  ("sr",  "Serbian"),   ("su",  "Sundanese"),
    ("sv",  "Swedish"),   ("sw",  "Swahili"),   ("ta",  "Tamil"),
    ("te",  "Telugu"),    ("tg",  "Tajik"),     ("th",  "Thai"),
    ("tk",  "Turkmen"),   ("tl",  "Tagalog"),   ("tr",  "Turkish"),
    ("tt",  "Tatar"),     ("uk",  "Ukrainian"), ("ur",  "Urdu"),
    ("uz",  "Uzbek"),     ("vi",  "Vietnamese"),("yi",  "Yiddish"),
    ("yo",  "Yoruba"),    ("yue", "Cantonese"), ("zh",  "Chinese"),
]
