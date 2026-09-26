"""Text normalisation, cross-script skeletons and tokenisation.

Everything here is pure Python + regex over individual records, so it is safe to
fan out over a process pool.  Two projections of every record are produced:

``name`` / ``address``
    The canonical, *native-script* form (case-folded, punctuation stripped,
    abbreviations expanded, legal suffixes removed).  These drive the
    string-similarity features.

``name_skel`` / ``addr_skel``
    A coarse *consonant skeleton*.  Roughly 4% of the true pairs are a Latin
    record matched against a Devanagari record (or vice versa); a skeleton makes
    the two scripts collide, e.g.::

        "राम मार्केटिंग"  ->  rm mrktng
        "Ram Marketing"   ->  rm mrktng

    The Devanagari table below is hand-written; no network call, gazetteer or
    third-party transliteration service is involved anywhere in this repository.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------------------
# Legal / organisational suffixes  (stripped from names and long address tokens)
# --------------------------------------------------------------------------------------
LEGAL_SUFFIXES = frozenset(
    """
    ltd limited pvt private priv inc incorporated incorp corp corporation
    llc llp lllp lp plc co company cos companies
    sa sas sarl nc gmbh bv nv ag kg ohg oy ab as oao ood spa
    pte pty enterprises enterprise ent holdings holding group
    """.split()
)

# Tokens that carry no identity information.
NAME_STOPWORDS = frozenset({"and", "the", "of", "for", "a", "an"})

# --------------------------------------------------------------------------------------
# Address abbreviation expansion
# --------------------------------------------------------------------------------------
ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "aven": "avenue", "blvd": "boulevard", "blv": "boulevard", "dr": "drive",
    "drv": "drive", "ln": "lane", "ct": "court", "crt": "court", "pl": "place",
    "plz": "plaza", "sq": "square", "hwy": "highway", "pkwy": "parkway",
    "pky": "parkway", "ter": "terrace", "terr": "terrace", "cir": "circle",
    "hts": "heights", "mt": "mount", "bldg": "building", "bldgs": "building",
    "fl": "floor", "flr": "floor", "apt": "apartment", "appt": "apartment",
    "ste": "suite", "no": "number", "nr": "number", "nos": "number",
    "numb": "number", "opp": "opposite", "near": "near", "behind": "behind",
    "beside": "beside", "ext": "extension", "grd": "ground", "lvl": "level",
    "k.no": "khasra", "kh.no": "khasra", "kh": "khasra", "s.no": "survey",
    "sno": "survey", "sec": "sector", "sect": "sector", "dist": "district",
    "distt": "district", "soc": "society", "nrl": "near", "gf": "ground",
    "1st": "1", "2nd": "2", "3rd": "3", "4th": "4", "5th": "5",
    "6th": "6", "7th": "7", "8th": "8", "9th": "9", "10th": "10",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
}

# Address tokens that are so common they make useless blocking keys; they are
# dropped from the *blocking* keys only and kept in the similarity text.
ADDRESS_BLOCK_STOPWORDS = frozenset(
    {
        "of", "and", "in", "at", "on", "near", "new", "old", "main", "post",
        "office", "number", "district", "state", "city", "town", "village",
        "opposite", "sector", "colony", "road", "street", "avenue", "lane",
        "north", "south", "east", "west", "nagar", "road.",
    }
)

# --------------------------------------------------------------------------------------
# Devanagari consonant skeleton
# --------------------------------------------------------------------------------------
_DEVA_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r",
    "ढ़": "rh", "फ़": "f", "य़": "y",
}
_DEVA_SKIPPED = frozenset(
    "ािीुूृेैोौॉॅॆॊॢऺऻऽऺ अआइईउऊऋएऐओऔऑऒऍऎ".replace(" ", "")
)
_DEVA_NASAL = {"ं": "n", "ँ": "n", "ऺ": "n", "ऻ": "n", "ः": "h", "ऽ": ""}
_DEVA_NUKTA = "़"
_DEVA_VIRAMA = "्"
_DEVA_DIGITS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}
_DEVA_SKEL_MAP = dict(_DEVA_CONSONANTS)
_DEVA_SKEL_MAP.update(_DEVA_NASAL)
_DEVA_SKEL_MAP.update(_DEVA_DIGITS)

_DEVA_CHARS = (
    set(_DEVA_CONSONANTS)
    | _DEVA_SKIPPED
    | set(_DEVA_NASAL)
    | set(_DEVA_DIGITS)
    | {_DEVA_VIRAMA, _DEVA_NUKTA}
)

# Accented Latin letters seen in transliterated Indian records.
_LATIN_FOLD = {
    "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a", "å": "a", "ā": "a",
    "é": "e", "è": "e", "ê": "e", "ë": "e", "ē": "e",
    "í": "i", "ì": "i", "î": "i", "ï": "i", "ī": "i",
    "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o", "ō": "o",
    "ú": "u", "ù": "u", "û": "u", "ü": "u", "ū": "u",
    "ñ": "n", "ń": "n", "ç": "c", "ş": "s", "ğ": "g",
    "ı": "i", "ł": "l", "ß": "ss", "æ": "ae", "œ": "oe",
    "’": "", "'": "", "`": "",
}
_LATIN_FOLD_KEYS = tuple(_LATIN_FOLD)
_LATIN_VOWELS = frozenset("aeiou")

# Python's ``\w`` excludes combining marks, which would shred every Devanagari
# syllable (the virama / matras are category Mn).  Widen the "keep" class to
# include the Devanagari and generic combining-mark blocks.
_KEEP = r"\w\u0900-\u097F\u0300-\u036F\u200C\u200D\u0E00-\u0E7F"
_RE_PUNCT = re.compile(r"[^" + _KEEP + r"]+", re.UNICODE)
_RE_DIGITS = re.compile(r"\d+")
_RE_DIGIT_RUN = re.compile(r"\d{3,}")


def has_deva(text: str) -> bool:
    for ch in text:
        if ch in _DEVA_CHARS:
            return True
    return False


def _fold_latin(text: str) -> str:
    for k in _LATIN_FOLD_KEYS:
        if k in text:
            text = text.replace(k, _LATIN_FOLD[k])
    return text


def latin_skeleton(text: str) -> str:
    """Drop Latin vowels, keeping consonant order and digits.

    ``"ram marketing private" -> "rmmrktngprvt"``
    """
    if not text:
        return ""
    return "".join(ch for ch in text if ch not in _LATIN_VOWELS)


def deva_skeleton(text: str) -> str:
    """Devanagari -> Latin consonant skeleton.

    ``"राम मार्केटिंग" -> "rm mrktng"``.  Vowels, matras, the virama and the nukta
    are dropped, so a cluster like "क्र" yields ``kr`` exactly like Latin "kr".
    """
    out = []
    ap = out.append
    for ch in text:
        if ch in _DEVA_SKEL_MAP:
            ap(_DEVA_SKEL_MAP[ch])
        elif ch in _DEVA_SKIPPED or ch == _DEVA_VIRAMA or ch == _DEVA_NUKTA:
            continue
        elif ch.isspace():
            ap(" ")
        elif ch.isdigit():
            ap(ch)
    return "".join(out)


def skeleton(text: str) -> str:
    """Cross-script-safe skeleton of an already-canonical string."""
    if not text:
        return ""
    return deva_skeleton(text) if has_deva(text) else latin_skeleton(text)


def _tokens(canonical: str, expand: bool, drop_legal: bool) -> list:
    toks = []
    for tok in _RE_PUNCT.sub(" ", canonical).split():
        if expand:
            tok = ADDRESS_ABBREV.get(tok, tok)
        if drop_legal and len(tok) > 1 and tok in LEGAL_SUFFIXES:
            continue
        if tok in NAME_STOPWORDS:
            continue
        if len(tok) == 1 and not tok.isdigit():
            continue
        toks.append(tok)
    return toks


def normalize(record_name, record_address):
    """Normalise one record into ``(name, address, name_skel, addr_skel)``."""
    record_name = "" if not record_name else str(record_name)
    record_address = "" if not record_address else str(record_address)

    n_raw = unicodedata.normalize("NFKC", record_name).casefold()
    name = " ".join(_tokens(_fold_latin(n_raw), expand=False, drop_legal=True))
    a_raw = unicodedata.normalize("NFKC", record_address).casefold()
    address = " ".join(_tokens(_fold_latin(a_raw), expand=True, drop_legal=False))

    n_skel_src = deva_skeleton(n_raw) if has_deva(n_raw) else _fold_latin(n_raw)
    a_skel_src = deva_skeleton(a_raw) if has_deva(a_raw) else _fold_latin(a_raw)
    name_skel = " ".join(t for t in latin_skeleton(n_skel_src).split() if t)
    addr_skel = " ".join(t for t in latin_skeleton(a_skel_src).split() if t)
    return name, address, name_skel, addr_skel


def normalize_pair_fast(record_name, record_address):
    """Just the similarity projection (skips skeleton work)."""
    return normalize(record_name, record_address)[:2]


def name_tokens(name: str) -> list:
    return [t for t in name.split() if t and t not in NAME_STOPWORDS and t not in LEGAL_SUFFIXES]


def address_tokens(address: str) -> list:
    return [t for t in address.split() if t and t not in NAME_STOPWORDS]


def digits_of(text: str) -> list:
    return _RE_DIGITS.findall(text)


def digit_runs_of(text: str) -> list:
    return _RE_DIGIT_RUN.findall(text)
