"""Tests for Devanagari -> Latin transliteration (team decision, 17 Aug 2026).

This table decides the spelling of every transcript in the ~4,000-clip Week-4
generation run, and the text is fed to XTTS rather than merely displayed. A wrong
vowel is a wrong *pronunciation*, repeated thousands of times and invisible in the
audio unless someone listens to that exact clip. So the rules are pinned here
rather than trusted.

Pure Python, no dependencies -- runs in CI, which installs only
ruff/pytest/numpy/pandas/pyyaml (P-004).
"""

import pandas as pd
import pytest

from src.data import transliteration as tr


# --------------------------------------------------------------------------- #
# Pass-through: a code-mixed corpus must keep its English exactly as written
# --------------------------------------------------------------------------- #
def test_pure_latin_text_is_returned_unchanged() -> None:
    text = "spoken tutorial project talktoateacher"
    assert tr.to_roman(text) == text


def test_transliteration_is_idempotent() -> None:
    # The column may be built twice, or over rows of mixed provenance.
    once = tr.to_roman("मुझे कल bank जाना है")
    assert tr.to_roman(once) == once


def test_english_words_inside_a_hindi_sentence_survive_verbatim() -> None:
    got = tr.to_roman("मैं इस image के इस भाग में zoom करना चाहती हूँ")
    assert "image" in got and "zoom" in got
    assert not tr.has_devanagari(got)


def test_punctuation_and_latin_digits_are_untouched() -> None:
    assert tr.to_roman("hello, world! 42") == "hello, world! 42"


def test_empty_string() -> None:
    assert tr.to_roman("") == ""


def test_non_string_input_is_coerced() -> None:
    assert tr.to_roman(42) == "42"


# --------------------------------------------------------------------------- #
# Output must contain no Devanagari at all -- the whole point of the exercise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "अच्छा अब यदि यह सफलतापूर्वक जुड़ता है",
        "एक table में विवरण प्रदर्शित होते हैं",
        "स्पोकन ट्यूटोरियल प्रोजेक्ट",
        "ऋषि ऐसे ही औरों को ॐ सिखाते",
    ],
)
def test_no_devanagari_survives(text: str) -> None:
    assert not tr.has_devanagari(tr.to_roman(text))


# --------------------------------------------------------------------------- #
# Schwa deletion -- the rule that makes it Hinglish rather than Sanskrit
# --------------------------------------------------------------------------- #
def test_word_final_schwa_is_dropped() -> None:
    # कमल is "kamal"; a scholarly scheme would write "kamala".
    assert tr.to_roman("कमल") == "kamal"


def test_medial_schwa_is_dropped_between_two_vowels() -> None:
    # An extra written vowel is an extra SYLLABLE to XTTS: "ka-ra-naa" vs
    # "kar-naa". This is a pronunciation bug, not a spelling preference.
    assert tr.to_roman("करना") == "karnaa"
    assert tr.to_roman("निकालने") == "nikaalne"
    assert tr.to_roman("चाहती") == "chaahtee"


def test_medial_schwa_is_kept_when_the_next_syllable_has_no_vowel() -> None:
    # कमल: dropping the final schwa leaves "l" vowel-less, so the medial "a"
    # must stay or the word collapses to "kaml".
    assert tr.to_roman("कमल") == "kamal"
    assert tr.to_roman("समझ") == "samajh"


def test_real_words_keep_their_syllable_count() -> None:
    for word, expected in (
        ("बचपन", "bachpan"),
        ("सरकार", "sarkaar"),
        ("नमस्ते", "namaste"),
    ):
        assert tr.to_roman(word) == expected, word


def test_a_single_syllable_keeps_its_vowel() -> None:
    # Dropping it would leave a bare consonant with no vowel at all.
    assert tr.to_roman("न") == "na"


def test_an_explicit_matra_is_never_deleted() -> None:
    # Only a schwa the parser supplied may go; a written vowel is real.
    assert tr.to_roman("जाना") == "jaanaa"


# --------------------------------------------------------------------------- #
# Consonants, conjuncts, and the borrowed nukta series
# --------------------------------------------------------------------------- #
def test_virama_joins_consonants_into_a_cluster() -> None:
    assert tr.to_roman("क्या") == "kyaa"


def test_nukta_consonants_use_their_borrowed_sounds() -> None:
    # Writing ज़ as "j" turns ज़रूरी into "jaroori" and loses the word.
    assert tr.to_roman("ज़रूरी") == "zarooree"


def test_decomposed_nukta_matches_the_precomposed_form() -> None:
    # ज़ exists both as one codepoint and as ज + nukta; MUCS contains both.
    assert tr.to_roman("ज़") == tr.to_roman("ज" + tr.NUKTA)


def test_aspirated_consonants_keep_their_h() -> None:
    assert tr.to_roman("अच्छा") == "achchhaa"


# --------------------------------------------------------------------------- #
# Nasals
# --------------------------------------------------------------------------- #
def test_anusvara_is_m_before_a_labial() -> None:
    assert tr.to_roman("संभव") == "sambhav"


def test_anusvara_is_n_elsewhere() -> None:
    assert tr.to_roman("हिंदी").startswith("hind")


def test_chandrabindu_becomes_a_nasal() -> None:
    assert "n" in tr.to_roman("हूँ")


# --------------------------------------------------------------------------- #
# Devanagari digits and danda
# --------------------------------------------------------------------------- #
def test_devanagari_digits_become_latin_digits() -> None:
    assert tr.to_roman("०१२३४५६७८९") == "0123456789"


def test_danda_becomes_a_full_stop() -> None:
    assert tr.to_roman("।") == "."


def test_danda_ends_a_word_without_swallowing_it() -> None:
    assert tr.to_roman("कमल।") == "kamal."


def test_visarga_stays_inside_its_word() -> None:
    # Treated as a separator it triggers the word-final schwa rule from mid-word,
    # turning अतः into "ath" instead of "atah".
    assert tr.to_roman("अतः") == "atah"
    assert tr.to_roman("दुःख") == "duhkh"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_marginal_letters_found_in_the_corpus_are_mapped() -> None:
    # A scan of all 56,143 indexed transcripts found exactly these two letters
    # uncovered; both would otherwise have leaked Devanagari into the corpus.
    assert tr.to_roman("ऱ") == "ra"
    assert tr.to_roman("ऩ") == "na"


def test_om_is_spelled_out() -> None:
    assert tr.to_roman("ॐ") == "om"


def test_an_unmapped_devanagari_sign_never_survives() -> None:
    # U+0951 is a vedic accent: no pronunciation, and it must not reach XTTS.
    assert not tr.has_devanagari(tr.to_roman("क॑मल"))


def test_unmapped_audit_is_empty_for_the_project_corpora() -> None:
    assert tr.unmapped_devanagari(["अच्छा", "करना", "hello", "ऱऩॐ"]) == {}


def test_unmapped_audit_reports_a_stray_sign() -> None:
    assert tr.unmapped_devanagari(["क॑"]) == {"॑": 1}


def test_has_devanagari_detects_a_single_embedded_character() -> None:
    assert tr.has_devanagari("hello क world")
    assert not tr.has_devanagari("hello world")


def test_romanise_series_maps_a_column() -> None:
    series = pd.Series(["कमल", "hello", "करना"])
    assert tr.romanise_series(series) == ["kamal", "hello", "karnaa"]


# --------------------------------------------------------------------------- #
# Determinism -- the reason this is hand-rolled rather than a dependency
# --------------------------------------------------------------------------- #
def test_output_is_stable_across_calls() -> None:
    text = "अच्छा अब यदि यह सफलतापूर्वक जुड़ता है तो बाकी स्क्रिप्ट रन होगी"
    assert len({tr.to_roman(text) for _ in range(10)}) == 1
