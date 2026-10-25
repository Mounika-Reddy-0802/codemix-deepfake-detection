"""The real phone-call test (week 14): scripts, cloning packs and playback files.

Three team members, each consenting in writing and with the mentor's approval, take
part in a small test over the real telephone network (``docs/real_calls/PROTOCOL.md``).
This module prepares everything that is not a phone call:

1. **Scripts.** Two call scripts of twelve code-mixed sentences (A: an everyday call,
   B: an urgent request for money and a code), plus four reference sentences each
   speaker reads once for cloning. Hindi is in Devanagari and English in Latin script,
   the MUCS convention; ``reading`` is a romanised copy for reading aloud. No sentence
   needs a gendered first-person verb, so every speaker can read every line.
2. **Packs.** ``pack`` turns each speaker's reference recording into a 16 kHz reference
   clip and writes one job table per tool, in the format ``src.data.generate_cli`` reads:
   XTTS-v2 (the seen tool, romanised text as in every earlier XTTS set) and Chatterbox
   (the hardest unseen tool, Devanagari text). Every row is in the ``realcall`` pool,
   which no training manifest may contain.
3. **Playback files.** ``calls`` joins each speaker's clones of one script into a single
   call of about a minute, and prepares the same speaker's own reading of that script
   the same way, so a cloned call and a genuine replayed call reach the phone through the
   same loudspeaker at the same level.

Run::

    python -m src.data.real_call_pack scripts --out docs/real_calls/SCRIPTS.md
    python -m src.data.real_call_pack pack --recordings data/real_calls/references \\
        --out data/real_calls/pack
    python -m src.data.real_call_pack calls --clones data/real_calls/clones \\
        --recordings data/real_calls/references --out data/real_calls/playback
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPEAKERS = ("P1", "P2", "P3")
POOL = "realcall"
TOOLS = ("xtts_v2", "chatterbox")
#: Pack folder per tool; the notebook finds packs by these folder names.
PACK_DIRS = {"xtts_v2": "rc_xtts", "chatterbox": "rc_chatterbox"}
REFERENCE_SR = 16_000
PLAYBACK_SR = 24_000
PLAYBACK_DBFS = -23.0
MIN_REFERENCE_SECONDS = 6.0
MAX_REFERENCE_SECONDS = 30.0
LEAD_SECONDS = 1.0
GAP_SECONDS = 0.5
#: XTTS truncates Hindi text longer than this without warning (P-015).
XTTS_MAX_CHARS = 150


@dataclass(frozen=True)
class Line:
    """One sentence of a script."""

    script: str  # "R" (reference), "A" or "B"
    number: int
    devanagari: str  # Hindi in Devanagari, English in Latin script
    reading: str  # romanised, for reading aloud

    @property
    def key(self) -> str:
        return f"{self.script}{self.number:02d}"


def _lines(script: str, pairs: list[tuple[str, str]]) -> tuple[Line, ...]:
    return tuple(Line(script, i, dev, rom) for i, (dev, rom) in enumerate(pairs, 1))


REFERENCE = _lines(
    "R",
    [
        (
            "आज का weather काफ़ी अच्छा है, शाम को हम park में walk के लिए जा सकते हैं.",
            "Aaj ka weather kaafi achha hai, shaam ko hum park mein walk ke liye ja sakte hain.",
        ),
        (
            "मेरे laptop की battery जल्दी खत्म हो जाती है, नया charger लेना पड़ेगा.",
            "Mere laptop ki battery jaldi khatam ho jaati hai, naya charger lena padega.",
        ),
        (
            "इस semester में हमारे पास तीन labs और दो projects हैं.",
            "Is semester mein hamaare paas teen labs aur do projects hain.",
        ),
        (
            "Library में नई books आई हैं, इस हफ़्ते जाकर देखनी चाहिए.",
            "Library mein nayi books aayi hain, is hafte jaakar dekhni chahiye.",
        ),
    ],
)

SCRIPT_A = _lines(
    "A",
    [
        (
            "हेलो, क्या अभी आपके पास two minutes हैं? मुझे एक important बात करनी है.",
            "Hello, kya abhi aapke paas two minutes hain? Mujhe ek important baat karni hai.",
        ),
        (
            "कल की meeting का time change हो गया है, अब वो शाम को five बजे होगी.",
            "Kal ki meeting ka time change ho gaya hai, ab woh shaam ko five baje hogi.",
        ),
        (
            "Project की final report हमें Monday तक submit करनी है.",
            "Project ki final report hamein Monday tak submit karni hai.",
        ),
        (
            "क्या आप मुझे presentation की slides WhatsApp पर भेज सकते हैं?",
            "Kya aap mujhe presentation ki slides WhatsApp par bhej sakte hain?",
        ),
        (
            "Lab में आज network बहुत slow था, इसलिए कुछ भी download नहीं हुआ.",
            "Lab mein aaj network bahut slow tha, isliye kuch bhi download nahi hua.",
        ),
        (
            "मुझे लगता है कि हमें results को एक बार और check करना चाहिए.",
            "Mujhe lagta hai ki hamein results ko ek baar aur check karna chahiye.",
        ),
        (
            "Canteen में आज बहुत भीड़ थी, तो हम लोग बाहर चले गए.",
            "Canteen mein aaj bahut bheed thi, toh hum log baahar chale gaye.",
        ),
        (
            "Exam का schedule notice board पर लग गया है, आप एक बार देख लेना.",
            "Exam ka schedule notice board par lag gaya hai, aap ek baar dekh lena.",
        ),
        (
            "अगर कोई problem हो तो मुझे directly call कर देना.",
            "Agar koi problem ho toh mujhe directly call kar dena.",
        ),
        (
            "Weekend पर घर जाने का plan है, ticket अभी book करनी है.",
            "Weekend par ghar jaane ka plan hai, ticket abhi book karni hai.",
        ),
        (
            "Mentor ने कहा है कि paper का draft जल्दी share करना है.",
            "Mentor ne kaha hai ki paper ka draft jaldi share karna hai.",
        ),
        (
            "ठीक है, बाकी बातें कल college में करते हैं, bye.",
            "Theek hai, baaki baatein kal college mein karte hain, bye.",
        ),
    ],
)

SCRIPT_B = _lines(
    "B",
    [
        (
            "हेलो, मेरा phone अभी खराब है, इसलिए दूसरे number से call किया है.",
            "Hello, mera phone abhi kharaab hai, isliye doosre number se call kiya hai.",
        ),
        (
            "मुझे एक urgent help चाहिए, क्या आप अभी free हैं?",
            "Mujhe ek urgent help chahiye, kya aap abhi free hain?",
        ),
        (
            "Hospital में थोड़ी emergency है और मेरे पास cash नहीं है.",
            "Hospital mein thodi emergency hai aur mere paas cash nahi hai.",
        ),
        (
            "क्या आप इस number पर five thousand rupees UPI कर सकते हैं?",
            "Kya aap is number par five thousand rupees UPI kar sakte hain?",
        ),
        (
            "शाम तक पूरा amount आपको वापस मिल जाएगा.",
            "Shaam tak poora amount aapko waapas mil jaayega.",
        ),
        (
            "आपके phone पर एक OTP आएगा, please वो मुझे बता देना.",
            "Aapke phone par ek OTP aayega, please woh mujhe bata dena.",
        ),
        (
            "Bank का app अभी open नहीं हो रहा है, इसलिए आपसे पूछना पड़ा.",
            "Bank ka app abhi open nahi ho raha hai, isliye aapse poochhna pada.",
        ),
        (
            "घर पर किसी को मत बताना, वो लोग बेकार में tension लेंगे.",
            "Ghar par kisi ko mat bataana, woh log bekaar mein tension lenge.",
        ),
        (
            "Payment होने के बाद मुझे screenshot भेज देना.",
            "Payment hone ke baad mujhe screenshot bhej dena.",
        ),
        (
            "Signal बहुत weak है, इसलिए आवाज़ थोड़ी अलग लग रही होगी.",
            "Signal bahut weak hai, isliye aawaaz thodi alag lag rahi hogi.",
        ),
        (
            "ये बात बहुत important है, please जल्दी कर देना.",
            "Yeh baat bahut important hai, please jaldi kar dena.",
        ),
        (
            "ठीक है, थोड़ी देर में फिर से बात करते हैं, thank you.",
            "Theek hai, thodi der mein phir se baat karte hain, thank you.",
        ),
    ],
)

SCRIPTS = {"A": SCRIPT_A, "B": SCRIPT_B}


def clip_name(tool: str, speaker: str, line: Line) -> str:
    """File name of one cloned sentence."""
    return f"{tool}_{speaker}_{line.key}.wav"


def playback_name(kind: str, speaker: str, script: str) -> str:
    """File name of one playback call: ``kind`` is ``genuine`` or a tool name."""
    return f"{kind}_{speaker}_{script}.wav"


def job_tables(speakers: tuple[str, ...] = SPEAKERS) -> dict[str, pd.DataFrame]:
    """One job table per tool, in the columns its driver reads, every row in ``POOL``."""
    from src.data.transliteration import to_roman

    reference_text = " ".join(line.devanagari for line in REFERENCE)
    xtts, chatterbox = [], []
    for speaker in speakers:
        for script, lines in SCRIPTS.items():
            for line in lines:
                xtts.append(
                    {
                        "job_id": len(xtts) + 1,
                        "cell": f"realcall_{script}",
                        "speaker": speaker,
                        "pool": POOL,
                        "reference_wav": f"refs/{speaker}.wav",
                        "transcript": to_roman(line.devanagari),
                        "transcript_source": line.devanagari,
                        "script_class": "mixed",
                        "language": "hi",
                        "tool": "xtts_v2",
                        "output_path": f"outputs/{clip_name('xtts_v2', speaker, line)}",
                        "seed": 11_000 + len(xtts) + 1,
                    }
                )
                chatterbox.append(
                    {
                        "job_id": len(chatterbox) + 1,
                        "speaker": speaker,
                        "pool": POOL,
                        "reference_wav": f"refs/{speaker}.wav",
                        "reference_text": reference_text,
                        "reference_utt_id": f"{speaker}_ref",
                        "transcript": line.devanagari,
                        "transcript_utt_id": line.key,
                        "language": "hi",
                        "tool": "chatterbox",
                        "output_path": clip_name("chatterbox", speaker, line),
                        "seed": 12_000 + len(chatterbox) + 1,
                    }
                )
    return {"xtts_v2": pd.DataFrame(xtts), "chatterbox": pd.DataFrame(chatterbox)}


def find_recording(folder: Path, stem: str) -> Path:
    """The one file named ``stem`` with any extension, e.g. ``P1_ref.m4a``."""
    found = sorted(p for p in folder.glob(f"{stem}.*") if p.is_file())
    if len(found) != 1:
        raise FileNotFoundError(f"expected exactly one {stem}.* in {folder}, found {found}")
    return found[0]


def prepare_reference(source: Path, target: Path) -> float:
    """Write a 16 kHz, level-normalised, trimmed reference clip; return its length (s)."""
    from src.utils.audio_utils import load_any, rms_normalize, save_wav, trim_silence

    audio = trim_silence(load_any(str(source), REFERENCE_SR), REFERENCE_SR)
    seconds = audio.size / REFERENCE_SR
    if seconds < MIN_REFERENCE_SECONDS:
        raise ValueError(
            f"{source.name}: {seconds:.1f} s of speech, need at least {MIN_REFERENCE_SECONDS:.0f} s"
        )
    audio = audio[: int(MAX_REFERENCE_SECONDS * REFERENCE_SR)]
    target.parent.mkdir(parents=True, exist_ok=True)
    save_wav(str(target), rms_normalize(audio, PLAYBACK_DBFS), REFERENCE_SR)
    return round(audio.size / REFERENCE_SR, 2)


def build_pack(recordings: Path, out: Path, speakers: tuple[str, ...] = SPEAKERS) -> pd.DataFrame:
    """One pack folder per tool (``refs/`` and ``generation_jobs.csv``); one row per speaker."""
    tables = job_tables(speakers)
    rows = []
    for speaker in speakers:
        source = find_recording(recordings, f"{speaker}_ref")
        for tool in TOOLS:
            seconds = prepare_reference(source, out / PACK_DIRS[tool] / "refs" / f"{speaker}.wav")
        rows.append({"speaker": speaker, "reference": source.name, "seconds": seconds})
    for tool, table in tables.items():
        table.to_csv(out / PACK_DIRS[tool] / "generation_jobs.csv", index=False)
    return pd.DataFrame(rows)


def join_clips(clips: list[np.ndarray], sr: int) -> np.ndarray:
    """Sentences end to end with a short pause, a lead-in and a tail, at the playback level."""
    from src.utils.audio_utils import rms_normalize

    lead = np.zeros(int(LEAD_SECONDS * sr), dtype=np.float32)
    gap = np.zeros(int(GAP_SECONDS * sr), dtype=np.float32)
    parts = [lead]
    for i, clip in enumerate(clips):
        parts += [clip.astype(np.float32)] if i == 0 else [gap, clip.astype(np.float32)]
    parts.append(lead)
    return rms_normalize(np.concatenate(parts), PLAYBACK_DBFS)


def assemble_calls(
    clones: Path, recordings: Path, out: Path, speakers: tuple[str, ...] = SPEAKERS
) -> pd.DataFrame:
    """Every playback call: clones of each script per tool, and each speaker's own reading."""
    from src.utils.audio_utils import load_any, rms_normalize, save_wav, trim_silence

    out.mkdir(parents=True, exist_ok=True)
    by_name = {p.name: p for p in clones.rglob("*.wav")}
    rows = []
    for speaker in speakers:
        for script, lines in SCRIPTS.items():
            for tool in TOOLS:
                names = [clip_name(tool, speaker, line) for line in lines]
                found = [by_name[n] for n in names if n in by_name]
                row = {"file": playback_name(tool, speaker, script), "kind": "clone", "tool": tool}
                row.update(speaker=speaker, script=script, clips=len(found))
                row["missing"] = " ".join(n for n in names if n not in by_name)
                if found:
                    audio = join_clips(
                        [trim_silence(load_any(str(p), PLAYBACK_SR), PLAYBACK_SR) for p in found],
                        PLAYBACK_SR,
                    )
                    save_wav(str(out / row["file"]), audio, PLAYBACK_SR)
                    row["seconds"] = round(audio.size / PLAYBACK_SR, 1)
                rows.append(row)
            source = find_recording(recordings, f"{speaker}_read_{script}")
            audio = trim_silence(load_any(str(source), PLAYBACK_SR), PLAYBACK_SR)
            pad = np.zeros(int(LEAD_SECONDS * PLAYBACK_SR), dtype=np.float32)
            audio = rms_normalize(np.concatenate([pad, audio, pad]), PLAYBACK_DBFS)
            name = playback_name("genuine", speaker, script)
            save_wav(str(out / name), audio, PLAYBACK_SR)
            rows.append(
                {
                    "file": name,
                    "kind": "genuine",
                    "tool": "none",
                    "speaker": speaker,
                    "script": script,
                    "clips": 1,
                    "missing": "",
                    "seconds": round(audio.size / PLAYBACK_SR, 1),
                }
            )
    return pd.DataFrame(rows)


def scripts_markdown() -> str:
    """The printable scripts: what each person reads, in both scripts."""
    out = [
        "# Real-call test: what to read",
        "",
        "Written by `python -m src.data.real_call_pack scripts`; do not edit by hand.",
        "Read the romanised line or the Devanagari line, whichever is easier. Speak at a",
        "normal pace, as on a phone call, and leave a short pause between sentences.",
        "",
    ]
    sections = [
        ("Reference (read once, for cloning: about 20 s)", REFERENCE),
        ("Script A: an everyday call (about 60 s)", SCRIPT_A),
        ("Script B: an urgent request (about 60 s)", SCRIPT_B),
    ]
    for title, lines in sections:
        out += [f"## {title}", "", "| # | Read this | Devanagari |", "|---|---|---|"]
        out += [f"| {line.key} | {line.reading} | {line.devanagari} |" for line in lines]
        out.append("")
    return "\n".join(out)


def main() -> None:
    """CLI: ``scripts``, ``pack`` or ``calls``."""
    import argparse

    parser = argparse.ArgumentParser(description="Real-call test: scripts, packs, playback")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scripts")
    sc.add_argument("--out", default="docs/real_calls/SCRIPTS.md")
    pk = sub.add_parser("pack")
    pk.add_argument("--recordings", required=True, help="folder with P1_ref.*, P2_ref.*, ...")
    pk.add_argument("--out", required=True)
    ca = sub.add_parser("calls")
    ca.add_argument("--clones", required=True, help="unzipped clone results (searched)")
    ca.add_argument("--recordings", required=True, help="folder with P1_read_A.*, ...")
    ca.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.cmd == "scripts":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(scripts_markdown() + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    elif args.cmd == "pack":
        print(build_pack(Path(args.recordings), Path(args.out)).to_string(index=False))
    else:
        table = assemble_calls(Path(args.clones), Path(args.recordings), Path(args.out))
        table.to_csv(Path(args.out) / "playback.csv", index=False)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
