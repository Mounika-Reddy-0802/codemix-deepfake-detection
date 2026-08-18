"""Devanagari -> Latin transliteration, so every transcript is readable Hinglish.

Team decision (17 Aug 2026): the project writes code-mixed text in **one script,
Latin**. Nobody on the team reads Devanagari, and a rater who cannot read the
target sentence cannot judge whether a clone said it -- they can only judge
whether it sounds pleasant, which is not what the pilot measures.

Filtering could not deliver this. MUCS is 98.2% Devanagari-bearing: of 52,825
transcripts only 957 contain no Devanagari at all, and those are plain English
sentences rather than Hinglish. Inside the frozen 25-speaker train pool exactly
17 usable Devanagari-free transcripts survive the length and digit filters. A
Latin-only corpus therefore has to be *produced*, not selected -- hence this
module.

**Why hand-rolled rather than `indic-transliteration`/`aksharamukha`.** Two
reasons, both load-bearing:

1. CI installs only ruff/pytest/numpy/pandas/pyyaml (P-004). A third-party
   transliterator would make every test here unrunnable in CI, which is where
   this table needs guarding most -- it is about to be applied to ~4,000
   generation transcripts.
2. The spelling has to be **frozen**. This output is not a display convenience;
   it is the text fed to XTTS. If a library upgrade changed one vowel, part of
   the corpus would be generated from one romanisation and part from another,
   and the difference would be invisible in the audio. A table in the repo
   changes only when someone edits it, in a reviewable diff.

**Target is Hinglish, not IAST.** The output is meant to look like how Hindi is
actually typed in Latin -- "mujhe kal bank jaana hai" -- so it uses digraphs
(aa, ee, oo, sh, ch) and no diacritics. Scholarly schemes romanise क as "ka" and
कमल as "kamala"; Hindi speakers write "kamal". The final-schwa rule below is what
produces that, and it is the single most visible difference between output that
reads as Hinglish and output that reads as Sanskrit.

Latin text, digits, and punctuation pass through untouched, so a code-mixed
sentence keeps its English words exactly as MUCS wrote them.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Devanagari tables
# --------------------------------------------------------------------------- #
VIRAMA = "्"  # halant: suppresses the inherent vowel
NUKTA = "़"  # dot below: forms the borrowed q/z/f series
ANUSVARA = "ं"
CHANDRABINDU = "ँ"
VISARGA = "ः"

#: Independent vowel letters (word-initial or standalone).
VOWELS = {
    "अ": "a",  # अ
    "आ": "aa",  # आ
    "इ": "i",  # इ
    "ई": "ee",  # ई
    "उ": "u",  # उ
    "ऊ": "oo",  # ऊ
    "ऋ": "ri",  # ऋ
    "ए": "e",  # ए
    "ऐ": "ai",  # ऐ
    "ओ": "o",  # ओ
    "औ": "au",  # औ
    "ऍ": "e",  # ऍ (borrowed, as in "cat")
    "ऑ": "o",  # ऑ (borrowed, as in "doctor")
}

#: Dependent vowel signs (matras), which replace a consonant's inherent "a".
MATRAS = {
    "ा": "aa",  # ा
    "ि": "i",  # ि
    "ी": "ee",  # ी
    "ु": "u",  # ु
    "ू": "oo",  # ू
    "ृ": "ri",  # ृ
    "े": "e",  # े
    "ै": "ai",  # ै
    "ो": "o",  # ो
    "ौ": "au",  # ौ
    "ॅ": "e",  # ॅ
    "ॉ": "o",  # ॉ
}

#: Consonants. Values exclude the inherent "a", which is added by the parser so
#: that the final-schwa rule can remove it again.
CONSONANTS = {
    "क": "k",  # क
    "ख": "kh",  # ख
    "ग": "g",  # ग
    "घ": "gh",  # घ
    "ङ": "ng",  # ङ
    "च": "ch",  # च
    "छ": "chh",  # छ
    "ज": "j",  # ज
    "झ": "jh",  # झ
    "ञ": "ny",  # ञ
    "ट": "t",  # ट
    "ठ": "th",  # ठ
    "ड": "d",  # ड
    "ढ": "dh",  # ढ
    "ण": "n",  # ण
    "त": "t",  # त
    "थ": "th",  # थ
    "द": "d",  # द
    "ध": "dh",  # ध
    "न": "n",  # न
    "प": "p",  # प
    "फ": "ph",  # फ
    "ब": "b",  # ब
    "भ": "bh",  # भ
    "म": "m",  # म
    "य": "y",  # य
    "र": "r",  # र
    "ल": "l",  # ल
    "ळ": "l",  # ळ
    "व": "v",  # व
    "श": "sh",  # श
    "ष": "sh",  # ष
    "स": "s",  # स
    "ह": "h",  # ह
    # Southern/marginal letters. Rare but real: a scan of all 56,143 indexed
    # transcripts found ऱ 103 times and ऩ 12 times, and without them those rows
    # would have carried Devanagari into the "Latin-only" corpus.
    "ऱ": "r",  # ऱ
    "ऩ": "n",  # ऩ
}

#: Nukta forms. Hindi borrows these for Perso-Arabic and English sounds; writing
#: ज़ as "j" instead of "z" turns "ज़रूरी" into "jaroori" and loses the word.
NUKTA_CONSONANTS = {
    "क़": "q",  # क़
    "ख़": "kh",  # ख़
    "ग़": "gh",  # ग़
    "ज़": "z",  # ज़
    "ड़": "r",  # ड़
    "ढ़": "rh",  # ढ़
    "फ़": "f",  # फ़
}

#: Devanagari digits. MUCS lecture speech contains them, and a stray ४ inside
#: otherwise-Latin text would reach XTTS as an unknown glyph.
DIGITS = {chr(0x0966 + n): str(n) for n in range(10)}

#: Punctuation and marks with no consonant/vowel value.
MARKS = {
    "।": ".",  # । danda -> full stop
    "॥": ".",  # ॥ double danda
    "ऽ": "",  # ऽ avagraha
    "ॐ": "om",  # ॐ
    "॰": ".",  # ॰ abbreviation sign
    VISARGA: "h",
}

#: Labials, before which an anusvara is pronounced "m" rather than "n"
#: (संभव -> "sambhav", not "sanbhav").
_LABIALS = frozenset("पफबभम")  # प फ ब भ म

#: Marks that end a word. Everything else in :data:`MARKS` sits *inside* one and
#: must stay in the run, or it triggers the word-final schwa rule from the middle
#: of a word: अतः read as "अत" + separator gives "ath" instead of "atah".
_SEPARATOR_MARKS = frozenset("।॥॰")  # । ॥ ॰

_DEVANAGARI_START, _DEVANAGARI_END = "ऀ", "ॿ"


def is_devanagari(char: str) -> bool:
    """Whether a character lies in the Devanagari block."""
    return _DEVANAGARI_START <= char <= _DEVANAGARI_END


def has_devanagari(text: str) -> bool:
    """Whether any character of ``text`` is Devanagari."""
    return any(is_devanagari(c) for c in str(text))


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def _units(word: str) -> list[tuple[str, str, bool]]:
    """Split a Devanagari run into ``(consonant, vowel, vowel_is_inherent)`` units.

    Consonant and vowel are kept apart because both schwa rules below delete a
    vowel while keeping its consonant. The flag marks a vowel the parser
    *supplied* rather than one the writer spelled with a matra -- only supplied
    schwas may be dropped, or "जाना" would lose its long aa and become "jaan".
    """
    units: list[tuple[str, str, bool]] = []
    i, n = 0, len(word)
    while i < n:
        char = word[i]

        # A consonant may be written as base + nukta or as its precomposed form.
        if i + 1 < n and word[i + 1] == NUKTA:
            composed = {
                "क": "q",
                "ख": "kh",
                "ग": "gh",
                "ज": "z",
                "ड": "r",
                "ढ": "rh",
                "फ": "f",
            }
            if char in composed:
                base, i = composed[char], i + 2
            else:  # nukta on a letter with no borrowed form: ignore the dot
                base, i = CONSONANTS.get(char, char), i + 2
        elif char in NUKTA_CONSONANTS:
            base, i = NUKTA_CONSONANTS[char], i + 1
        elif char in CONSONANTS:
            base, i = CONSONANTS[char], i + 1
        elif char in VOWELS:
            units.append(("", VOWELS[char], False))
            i += 1
            continue
        elif char in DIGITS:
            units.append((DIGITS[char], "", False))
            i += 1
            continue
        elif char in MARKS:
            units.append((MARKS[char], "", False))
            i += 1
            continue
        elif char in (ANUSVARA, CHANDRABINDU):
            # "m" before a labial, "n" everywhere else.
            nxt = word[i + 1] if i + 1 < n else ""
            units.append(("m" if nxt in _LABIALS else "n", "", False))
            i += 1
            continue
        elif char == VIRAMA:
            i += 1  # a virama with no consonant before it carries no sound
            continue
        elif is_devanagari(char):
            # Safety net. The corpus scan that built these tables found only two
            # uncovered letters (ऱ, ऩ), both now mapped -- but the guarantee this
            # module sells is "no Devanagari reaches XTTS", and an unmapped sign
            # in a corpus nobody has scanned yet would break it silently. Vedic
            # accents and editorial marks carry no pronunciation, so dropping is
            # the right default; `unmapped_devanagari` reports them for review.
            i += 1
            continue
        else:
            units.append((char, "", False))
            i += 1
            continue

        # The consonant is placed; whatever follows decides its vowel.
        if i < n and word[i] == VIRAMA:
            units.append((base, "", False))  # bare consonant, joins the next one
            i += 1
        elif i < n and word[i] in MATRAS:
            units.append((base, MATRAS[word[i]], False))
            i += 1
        else:
            units.append((base, "a", True))  # inherent schwa, possibly dropped
    return units


def _delete_schwas(units: list[tuple[str, str, bool]]) -> str:
    """Apply Hindi schwa deletion -- the rule that turns Sanskrit into Hinglish.

    Two rules, applied right to left because the final one feeds the medial one:

    **Final.** Hindi does not pronounce a word's inherent last vowel: कमल is
    "kamal", not "kamala".

    **Medial.** An inherent schwa also drops between two vowel-bearing syllables
    (the classic ``VC_CV`` environment): करना is "karnaa", not "karanaa", and
    निकालने is "nikaalne", not "nikaalane". This is not cosmetic. The text goes
    to XTTS, and an extra written vowel is an extra *syllable* -- "ka-ra-naa"
    instead of "kar-naa" -- so skipping this rule would mispronounce a large
    fraction of every generated clip.

    Right-to-left with an immediate re-check stops two adjacent schwas from both
    dropping, which would strand a consonant cluster with no vowel at all.
    """
    if not units:
        return ""
    vowels = [vowel for _, vowel, _ in units]
    inherent = [flag for _, _, flag in units]

    # Final schwa. Skipped on a one-unit word, where it would leave a bare
    # consonant ("na" -> "n").
    if len(units) > 1 and inherent[-1]:
        vowels[-1] = ""

    # Medial schwa: drop unit i when its neighbours both still carry a vowel.
    for i in range(len(units) - 2, 0, -1):
        if not inherent[i] or not vowels[i]:
            continue
        if vowels[i - 1] and vowels[i + 1]:
            vowels[i] = ""

    return "".join(cons + vowel for (cons, _, _), vowel in zip(units, vowels, strict=True))


def to_roman(text: str) -> str:
    """Transliterate the Devanagari in ``text`` to Latin, leaving the rest alone.

    Idempotent on text that is already Latin, so it is safe to apply to a
    code-mixed corpus twice or to a column of mixed provenance.
    """
    text = str(text)
    if not has_devanagari(text):
        return text

    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            out.append(_delete_schwas(_units("".join(run))))
            run.clear()

    for char in text:
        # Digits and sentence separators end a word; visarga, avagraha and ॐ do
        # not, so they stay in the run and are resolved by `_units`.
        if is_devanagari(char) and char not in DIGITS and char not in _SEPARATOR_MARKS:
            run.append(char)
        else:
            flush()
            out.append(DIGITS.get(char) or MARKS.get(char) or char)
    flush()
    return "".join(out)


def romanise_series(values) -> list[str]:
    """Transliterate an iterable of transcripts. Convenience for pandas columns."""
    return [to_roman(v) for v in values]


def unmapped_devanagari(values) -> dict[str, int]:
    """Devanagari characters the tables do not cover, with their counts.

    The audit to run before pointing this module at a new corpus: anything
    returned here is being silently dropped by the safety net in :func:`_units`,
    which is correct for editorial marks and wrong for a letter. Returned empty
    for MUCS and HiACC as of 17 Aug 2026.
    """
    known = (
        set(VOWELS)
        | set(MATRAS)
        | set(CONSONANTS)
        | set(NUKTA_CONSONANTS)
        | set(DIGITS)
        | set(MARKS)
        | {VIRAMA, NUKTA, ANUSVARA, CHANDRABINDU}
    )
    counts: dict[str, int] = {}
    for value in values:
        for char in str(value):
            if is_devanagari(char) and char not in known:
                counts[char] = counts.get(char, 0) + 1
    return counts
