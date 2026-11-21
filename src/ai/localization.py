"""Deterministic normalization and script checks for localized AI output."""

from __future__ import annotations

import re

from opencc import OpenCC


_TRADITIONAL_TO_SIMPLIFIED = OpenCC("t2s")

_HANGUL_RE = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]")
_CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_LETTER_RE = re.compile(
    r"[A-Za-z\u0400-\u04FF\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3"
    r"\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]"
)


def normalize_language(text: str, language: str) -> str:
    """Normalize generated text to the script implied by its language tag."""
    if language.lower() == "zh":
        return _TRADITIONAL_TO_SIMPLIFIED.convert(text)
    return text


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text or ""))


def script_counts(text: str) -> dict[str, int]:
    return {
        "letters": _count(_LETTER_RE, text),
        "hangul": _count(_HANGUL_RE, text),
        "cjk": _count(_CJK_RE, text),
        "cyrillic": _count(_CYRILLIC_RE, text),
        "latin": _count(_LATIN_RE, text),
    }


def assert_matches_language(text: str, language: str, *, label: str = "text") -> None:
    """Reject obvious wrong-script output for the requested language tag.

    Proper nouns and short Latin tokens are allowed. A body that is mostly Hangul
    or CJK while the digest asks for Russian/English is not.
    """
    counts = script_counts(text)
    letters = counts["letters"]
    if letters < 12:
        return

    hangul = counts["hangul"]
    cjk = counts["cjk"]
    cyrillic = counts["cyrillic"]
    latin = counts["latin"]
    foreign_east = hangul + cjk
    lang = (language or "").lower()

    if lang in {"ru", "uk", "bg"}:
        if foreign_east >= 12 and foreign_east > cyrillic:
            raise ValueError(
                f"{label} is not in {language}: hangul/cjk dominate Cyrillic"
            )
        return

    if lang in {"en", "de", "fr", "es", "it", "pt", "nl", "pl", "cs", "sv", "no", "da", "fi"}:
        if foreign_east >= 12 and foreign_east > latin:
            raise ValueError(
                f"{label} is not in {language}: hangul/cjk dominate Latin"
            )
        return

    if lang.startswith("zh") or lang in {"ja", "ko"}:
        return

    # Unknown tags: still block Hangul-heavy output when Latin/Cyrillic was expected
    # by a non-CJK language code that we did not list above.
    if foreign_east >= 24 and foreign_east > max(latin, cyrillic) * 2:
        raise ValueError(f"{label} looks untranslated for language `{language}`")
