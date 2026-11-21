import pytest

from src.ai.localization import assert_matches_language, normalize_language, script_counts


def test_normalize_language_passes_russian_through():
    text = "KDDI завершила IoT-тестирование"
    assert normalize_language(text, "ru") == text


def test_script_counts_detect_hangul():
    counts = script_counts(
        "IoT 개발과 한국 KITEKI 인증: 모듈·안테나 세트, 케이스 포함 인증의 함정과 확인 난점"
    )
    assert counts["hangul"] >= 20
    assert counts["cyrillic"] == 0


def test_russian_rejects_hangul_body():
    korean = (
        "IoT 디바이스를 현장에 배포하려는 개발자가 한국의 기술기준적합증명(KITEKI)에서 "
        "직면한 복잡함을 설명합니다. 모듈 자체보다는 모듈+안테나 세트로 인증되며, "
        "인증되지 않은 안테나 교체는 전파법 위반이 됩니다."
    )
    with pytest.raises(ValueError, match="hangul/cjk"):
        assert_matches_language(korean, "ru", label="title")


def test_russian_allows_latin_brands_with_cyrillic_body():
    text = (
        "KDDI завершила межсетевое тестирование телефона INFOBAR ZERO и официально "
        "подтвердила его совместимость и гарантированную работу в сети au."
    )
    assert_matches_language(text, "ru", label="title")


def test_english_rejects_hangul_body():
    korean = (
        "IoT 디바이스를 현장에 배포하려는 개발자가 한국의 기술기준적합증명에서 "
        "직면한 복잡함을 설명합니다. 모듈과 안테나 세트 인증이 필요합니다."
    )
    with pytest.raises(ValueError, match="hangul/cjk"):
        assert_matches_language(korean, "en", label="block summary")
