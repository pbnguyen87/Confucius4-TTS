"""Vietnamese text normalization for TTS -- a pure-Python re-implementation of vinorm.

Why not vinorm itself: the package ships a prebuilt Linux x86 binary (``vinorm/main``)
and calls it through a subprocess. On macOS, ARM, or any other host it fails with
``OSError: [Errno 8] Exec format error``, so output silently differed between
machines. This module reimplements vinorm's rule set in Python and reuses its
dictionaries, vendored as TSV under ``frontend/data/``. There is no vinorm import.

Rules ported, one per file in vinorm's ``RegexRule/``:

====================  ===========================================================
Website, Email        protected verbatim (digits inside must not be read)
Date_1                dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy
Date_3                context word + dd/mm with no year ("ngày 30/4")
Date_From_To_1/2      "từ 20-25/3", "từ 01-05/2020"
Month                 "tháng 9/2026"
Time                  "9h30", "9h", "9:30", with am/pm
PhoneNumber           mobile/landline plus 1800/1900 hotlines, digit by digit
FootballUnder         "U23", "U.23"
FootballOther         "tỉ số 3-1", "đội hình 4-4-2"
PoliticalDivision     "Q.1", "P.5", "H.Bình Chánh", "TX.", "TP.", "X.", "KP."
RomanNumber           "thứ III", "kỷ XX", "khóa IV"
Measurement           number + unit, including ratios ("5 ng/ml", "100 km/h")
Measurement_1         number + currency or percent symbol ("5$", "50%")
Street, Office        "đường/số/phòng/lớp" + code token, read character by character
NormalNumber          dot-grouped, comma-grouped, decimal comma, decimal dot
Codenumber            catch-all code reading (opt-in, ``read_codes``)
====================  ===========================================================

Tables ported from ``Mapping/`` and ``Dict/``: Acronyms (3.6k), Acronyms_shorten
(3.1k), Teencode (484), BaseUnit, CurrencyUnit, PrefixUnit, Symbol, LetterSoundVN,
LetterSoundEN, Number, and the Popular Vietnamese word list (7.7k), used to decide
whether an unknown token is a real word.

Two deliberate departures from vinorm, both measured on this project's corpora
(ViMedCSS + UIT-ViSFD, 26,940 sentences):

* **Acronyms expand conservatively.** vinorm expands every token in its table; that
  rewrote 2,557 tokens here, many wrongly ("mg" -> "mật gấu", "sp" -> "sư phạm").
  See :func:`_expand_acronym`.
* **Letter-by-letter reading of unknown acronyms is opt-in** (``spell_acronyms``).
  Spelling "GPT" as "giê bê tê" destroys the English spans the code-switching
  benchmark scores against.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# --------------------------------------------------------------------------- tables
_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_tsv(name: str) -> dict[str, str]:
    m: dict[str, str] = {}
    try:
        for line in (_DATA_DIR / name).read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                k, v = k.strip(), v.strip()
                if k and v:
                    m.setdefault(k, v)
    except OSError:
        pass
    return m


def _load_set(name: str) -> set[str]:
    try:
        return {w.strip().lower() for w in (_DATA_DIR / name).read_text(encoding="utf-8").split() if w.strip()}
    except OSError:
        return set()


_NUMBER_WORDS = _load_tsv("number.tsv")
_DIGITS = [_NUMBER_WORDS.get(str(i)) or d for i, d in enumerate(
    ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"])]

_UNIT_MAP: dict[str, str] = {**_load_tsv("base_unit.tsv"), **_load_tsv("currency.tsv")}
_UNIT_MAP.update({
    "c": "độ xê", "°c": "độ xê", "kwh": "ki lô oát giờ",
    "km/h": "ki lô mét một giờ", "km/s": "ki lô mét một giây",
    "m2": "mét vuông", "m3": "mét khối",
    # clinical and electrical units the vinorm table lacks
    "ng": "nanogam", "mcg": "micrôgam", "µg": "micrôgam",
    "mmol": "mi li mon", "dl": "đê xi lít", "iu": "đơn vị quốc tế",
    "v": "vôn", "hz": "héc", "db": "đê xi ben",
    "mah": "mi li am pe giờ", "wh": "oát giờ", "va": "vôn am pe",
    "\u2103": "độ xê", "\u2109": "độ ép",
})
_CURRENCY = _load_tsv("currency.tsv")          # % $ € £ ¥ ₩ ₭ ₱ ฿ Ω
_PREFIX_UNIT = _load_tsv("prefix_unit.tsv")    # SI prefixes: d c m μ n p f a ...
_SYMBOLS: dict[str, str] = _load_tsv("symbol.tsv")
_SYMBOLS["/"] = "trên"   # vinorm says "xuyệt"; "trên" reads naturally in ratios
_SYMBOLS.setdefault("\u2030", "phần nghìn")      # per-mille
_SYMBOLS.setdefault("\u2103", "độ xê")           # single-codepoint degree Celsius
_SYMBOLS.setdefault("\u2109", "độ ép")
_FRACTION_CHARS = "\u00bc\u00bd\u00be\u2189" + "".join(chr(c) for c in range(0x2150, 0x215F))
_TEENCODE = _load_tsv("teencode.tsv")
# vinorm's chat table collides with common English words. "net" -> "nét" and "hot" ->
# "hót" are the right Vietnamese loanword readings and are kept; "won" -> "quan" would
# mangle English past tense in code-switched text.
# Chat-shorthand keys that are ordinary English words. Vietnamese loanword readings
# that happen to be English words are kept on purpose ("net" -> "nét", "hot" -> "hót",
# "chip" -> "chíp", "oxy" -> "ô xi"), and "ko" -> "không" is the commonest shorthand
# there is. The rest only ever fired inside English spans.
for _k in ("won", "ah", "oh", "lot", "lip", "fog", "slay", "hut", "ting", "yang",
           "jan", "bing", "gar", "mar", "lar", "kor", "sur", "rah", "rit", "roy",
           "zo", "ge", "blo", "pang", "prong", "sara", "maha", "khmer", "lich",
           "dap", "dak", "trog", "trug", "yah", "plup", "sok", "lak", "kra", "eli",
           "bobo", "chama", "nim"):
    _TEENCODE.pop(_k, None)
_LETTER_VN = _load_tsv("letter_sound_vn.tsv")
_LETTER_EN = _load_tsv("letter_sound_en.tsv")
_POPULAR = _load_set("popular_vi.txt")
_ACRONYMS: dict[str, str] = {**_load_tsv("acronyms_shorten.tsv"), **_load_tsv("acronyms.tsv")}

# Single upper-case letters that follow a digit but are not units -- see _unit_sub.
_NON_UNIT_UPPER = {"G", "M", "K", "B", "T", "E", "C"}
_ACRONYM_LOWER_ALLOW = {"đt", "tgdđ", "ntn", "qtv"}
# Lower-case keys of the acronym table that are also ordinary English words; checked
# against the 234,456-word system dictionary, these two are the only collisions.
_ACRONYM_EN_BLOCK = {"cand", "telt"}
# Table entries wrong in a technical / code-switched context, or too ambiguous
# ("CT" is a CT scan in medical speech, not "chương trình").
_ACRONYM_SKIP = {"pc", "tb", "ct"}
# vinorm's reading is the academic sense; in consumer and clinical Vietnamese --
# the registers in this project's data -- these mean something else.
_ACRONYM_OVERRIDE = {
    "nv": "nhân viên", "sp": "sản phẩm", "ch": "cửa hàng", "kh": "khách hàng",
    # Retail senses, read off the contexts these appear in ("đi bảo hành ... ttbh đưa
    # máy về", "nhân viên của tgdt", "ghé TTDD ngã 4 Âu Cơ"). The table's academic
    # readings -- "thị trấn bỏ hoang", "tôn giáo dân tộc", "trị giá giao dịch" -- are
    # plainly wrong there; tgdt and tggd are keyboard variants of tgdđ.
    "ttbh": "trung tâm bảo hành", "ttdd": "trung tâm di động",
    "tgdt": "thế giới di động", "tggd": "thế giới di động",
}
# Two- and three-letter upper-case tokens are where the table collides with English
# and scientific abbreviations (HCV -> "huy chương vàng", VP -> "văn phòng", NMN ->
# "nhà máy nước"). Four letters and up are safe, so short tokens expand only here.
_ACRONYM_SHORT_ALLOW = {"vn", "tp", "đt", "tv", "nv", "sp", "ch", "kh", "hn"}

ABBREVIATIONS = {
    r"\bTP\.?\s*HCM\b": "thành phố Hồ Chí Minh",
    r"\bHN\b": "Hà Nội",
    r"\bUBND\b": "ủy ban nhân dân",
    r"\bVN\b": "Việt Nam",
    r"\bTS\.": "tiến sĩ ",
    r"\bThS\.": "thạc sĩ ",
    r"\bGS\.": "giáo sư ",
    r"\bPGS\.": "phó giáo sư ",
    r"\bBS\.": "bác sĩ ",
    r"\bv\.v\.": "vân vân ",
    r"\bvs\b": "với",
    r"\bUSD\b": "đô la Mỹ",
    r"\bVNĐ\b|\bVND\b|\bđ\b": "đồng",
}

# --------------------------------------------------------------------------- patterns
# Invisible characters that must never reach the model: C0 controls, DEL,
# zero-width space/joiners, line/paragraph separators, BOM.
# An invisible character BETWEEN TWO ALPHANUMERICS is a separator, not noise. Content
# management systems insert zero-width spaces at line-break opportunities, and deleting
# one welds two words into a single token ("tiếng<ZWSP>Việt" -> "tiếngViệt") or, between
# digits, invents a different number ("12<ZWSP>3" -> 123). Anywhere else -- in
# particular before a combining mark, where a space would block composition -- it is
# dropped. A soft hyphen is deliberately absent here: it marks a hyphenation point
# inside one word, so it is always deleted.
_ZW_SEPARATORS = "\u200b\u200c\u200d\u2060\ufeff"
_ZW_SEP_RE = re.compile(rf"(?<=[^\W_])[{_ZW_SEPARATORS}]+(?=[^\W_])")
# U+2028 and U+2029 are line and paragraph separators: whitespace, not noise.
_LINE_SEP_RE = re.compile("[\u2028\u2029]+")
_CONTROL_RE = re.compile(
    "[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u00ad\u180e"
    "\u200b-\u200f\u2028\u2029\u2060-\u2064\ufeff]"
)
# Characters Unicode files as numbers but not as digits: superscripts, subscripts,
# circled numerals, Roman numeral letterforms, vulgar fractions. Python's "\w" counts
# them as word characters, which made "gah½" a single token and let a rule fire only
# on a second pass; they must never be part of a word.
_NUMERIC_NONWORD = ("\u00b2\u00b3\u00b9\u00bc-\u00be\u2070-\u209f"
                    "\u2150-\u218f\u2460-\u24ff")
_VI_WORD = rf"[^\W\d_{_NUMERIC_NONWORD}]"

# "À-Ỹ" looks like an upper-case range but spans both cases: "à" (U+00E0) and "ị"
# (U+1ECB) sit inside it, so [A-ZÀ-Ỹ] matched lower-case letters too and every rule
# built on it mis-fired. These classes are derived from Unicode case instead.
_UPPER_CLS = "A-Z" + "".join(chr(c) for c in range(0x00C0, 0x1EFA) if chr(c).isupper())
_LOWER_CLS = "a-z" + "".join(chr(c) for c in range(0x00C0, 0x1EFA) if chr(c).islower())


def _lookalike_table() -> dict[int, str]:
    """Fold numeric look-alikes onto ASCII so the ordinary rules can read them.

    "25m²" is how Vietnamese writes square metres; without this the superscript
    survives every rule and the model is handed "mét²". Vulgar fractions are left
    alone -- they are read as words by the symbol table, not as digits.
    """
    m: dict[str, str] = {}
    for i, ch in enumerate("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"):
        m[ch] = str(i)
    for i in range(20):
        m[chr(0x2460 + i)] = f" {i + 1} "               # circled numerals stand alone
    for cp in range(0x2160, 0x2180):                     # Roman numeral letterforms
        d = unicodedata.normalize("NFKD", chr(cp))
        if d.isalpha():
            m[chr(cp)] = d
    return str.maketrans(m)


_LOOKALIKE = _lookalike_table()
_SUP_DIGITS = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
_SUPERSCRIPT_RE = re.compile(f"[{_SUP_DIGITS}]+")


def _superscript_sub(m: re.Match) -> str:
    """"m²" is an exponent the unit table knows; "10³" is a power of ten.

    Folding every superscript onto a plain digit turned "10³" into "một trăm lẻ ba",
    so the digit is only folded when a letter precedes it.
    """
    digits = "".join(str(_SUP_DIGITS.index(c)) for c in m.group(0))
    prev = m.string[m.start() - 1] if m.start() else ""
    if prev.isalpha():
        return digits
    return f" mũ {number_to_vietnamese(int(digits))} "

# Decoration, not speech: reading these aloud or leaving them for the model to
# vocalise both sound wrong, so "Samsung(R)" simply becomes "Samsung".
_DROP_RE = re.compile("[\u2122\u00ae\u2120\u2117]")


def _vulgar_fractions() -> dict[str, str]:
    """A vulgar fraction is spoken, not drawn: "1/2 vien" must not lose the half.
    Every such character decomposes to "n FRACTION SLASH d", so the table is derived
    from Unicode rather than hand-written."""
    out: dict[str, str] = {}
    for cp in [0xBC, 0xBD, 0xBE, 0x2189, *range(0x2150, 0x215F)]:
        ch = chr(cp)
        parts = unicodedata.normalize("NFKD", ch).split("\u2044")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            den = int(parts[1])
            out[ch] = (f"{number_to_vietnamese(int(parts[0]))} phần "
                       f"{'tư' if den == 4 else number_to_vietnamese(den)}")
    return out

# Website / Email (vinorm Website.txt, Email.txt)
# The local-part class is ASCII, so without a left boundary the match could begin in
# the middle of a Vietnamese word and tear it apart ("hộpthưtuan@gmail.com").
_EMAIL_RE = re.compile(
    r"(?<![^\W\d_])[A-Za-z0-9._%+'!#$&*=?^`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_URL_RE = re.compile(
    r"(?:https?://|ftp://|sftp://|www\.)[^\s<>\"']+"
    r"|\b[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*"
    r"\.(?:com|net|org|info|coop|int|edu|gov|io|vn|uk)"
    r"(?:[./][^\s]*)?"
)

# Dates (Date_1, Date_3, Date_From_To_1/2, Month)
# The middle field is validated in the handler rather than by the pattern, so a
# US-order date ("12/25/2020") can be recognised and swapped instead of falling
# through to a chain of "trên".
_DATE_FULL_RE = re.compile(r"\b([0-3]?\d)\s?[/.-]\s?([0-3]?\d)\s?[/.-]\s?([12]\d{3})\b")
_DATE_CTX = r"(ngày|sáng|trưa|chiều|tối|đêm|hôm|mùng|mồng|lúc|từ|đến)"
_DATE_DM_RE = re.compile(rf"\b{_DATE_CTX}\s+([0-3]?\d)\s?[/.-]\s?([01]?\d)\b", re.IGNORECASE)
_DATE_RANGE_DD_RE = re.compile(r"\b(từ|ngày)\s+([0-3]?\d)\s?-\s?([0-3]?\d)\s?[/.]\s?([01]?\d)\b", re.IGNORECASE)
_DATE_RANGE_DMDM_RE = re.compile(
    r"\b(từ|ngày)\s+([0-3]?\d)[/.]([01]?\d)\s?(?:-|đến)\s?([0-3]?\d)[/.]([01]?\d)\b", re.IGNORECASE)
# "Thời gian 11/6 đến 11/7" read the first pair as a ratio and the second as a date,
# because only the second had a context word in front of it. Two d/m pairs joined by
# "đến" or a dash are a date range on their own -- but "uống 1/2 đến 3/4 viên" has the
# same shape, so when both day parts could be months the sentence must carry a date cue.
_DATE_RANGE_BARE_RE = re.compile(
    r"(?<![\d/])([0-3]?\d)\s?/\s?([01]?\d)\s*(?:đến|-|\u2013)\s*"
    r"([0-3]?\d)\s?/\s?([01]?\d)(?![\d/])")
_DATE_CUE_RE = re.compile(
    r"(?i)\b(thời gian|thời hạn|thời điểm|hạn|ngày|lịch|diễn ra|khai mạc|kéo dài|"
    r"dự kiến|áp dụng|từ)\b")


def _date_range_bare_sub(m: re.Match) -> str:
    d1, m1, d2, m2 = (int(x) for x in m.groups())
    if not (1 <= m1 <= 12 and 1 <= m2 <= 12 and 1 <= d1 <= 31 and 1 <= d2 <= 31):
        return m.group(0)
    if max(d1, d2) <= 12 and not _DATE_CUE_RE.search(
            m.string[max(0, m.start() - 40): m.start()]):
        # Ambiguous ("uống 1/2 đến 3/4 viên"): read both sides as ratios here rather
        # than leaving them, because "đến" counts as a date context word and the
        # day-month rule would otherwise claim the second pair alone and read one
        # side as a fraction and the other as a date.
        return f" {_frac_words(d1, m1)} đến {_frac_words(d2, m2)} "
    return (f"ngày {number_to_vietnamese(d1)} tháng {number_to_vietnamese(m1)} "
            f"đến ngày {number_to_vietnamese(d2)} tháng {number_to_vietnamese(m2)}")


_MONTH_YEAR_RE = re.compile(r"\btháng\s+([01]?\d)\s?[/.-]\s?([12]\d{3})\b", re.IGNORECASE)
_MONTH_RANGE_RE = re.compile(r"\b(từ|tháng)\s+([01]?\d)\s?-\s?([01]?\d)[/.]([12]\d{3})\b", re.IGNORECASE)

# Time (Time.txt)
# "(?:\s?([ap]m))?" keeps the optional space INSIDE the group: a bare "\s?" ate the
# space before the next word and welded it ("9h30 sáng" -> "... phútsáng").
_TIME_HM_RE = re.compile(
    # The minute marker is consumed so the handler cannot emit "phút" twice. "(?!\d)"
    # replaces a word boundary, which would not fit between "30" and the "m" of
    # "1h30m"; the trailing guard keeps "9h30 mai" from losing its "m".
    r"\b(2[0-4]|[01]?\d)\s?[:hg]\s?([0-5]\d)(?!\d)(?:\s?([ap]m))?"
    r"(?:\s*(?:phút|ph|m)(?![A-Za-zÀ-ỹ0-9]))?",
    re.IGNORECASE,
)
_TIME_H_RE = re.compile(r"\b(2[0-4]|[01]?\d)\s?h\b(?!\w)")
# vinorm also accepts "8g" for eight o'clock, but "g" is grams in every recipe,
# nutrition label and dosage, so the clock reading needs a cue word beside it.
_TIME_G_RE = re.compile(r"\b(2[0-4]|[01]?\d)\s?g\b(?!\w)")
_CLOCK_BEFORE_RE = re.compile(r"(?i)(l[úu]c|kho[ảa]ng|t[ừu]|đ[ếe]n|h[ồo]i|t[ầa]m|tr[ưu][ớo]c|sau)\s*$")
_CLOCK_AFTER_RE = re.compile(r"(?i)^\s*(s[áa]ng|tr[ưu]a|chi[ềe]u|t[ốo]i|đ[êe]m|khuya)")
# "mạng 4g" is a network generation, not four grams.
_NETWORK_BEFORE_RE = re.compile(
    r"(?i)(m[ạa]ng|sim|s[óo]ng|g[óo]i|wifi|data)\s*\d{0,2}\s*$")
_TIME_AMPM_RE = re.compile(r"\b(1[0-2]|[1-9])\s?([ap])\.?m\b\.?", re.IGNORECASE)

# Phone (PhoneNumber.txt)
_HOTLINE_RE = re.compile(r"(?<![\w.])1[89]00[\s.]?[\d\s.]{4,8}\d(?![\w.])")
# Separators may include the parentheses of an area code: "(024) 3825-1234".
_PHONE_SEP = r"[-\s.()]{0,3}"
_PHONE_RE = re.compile(
    rf"(?<![\w.])((?:\+\d{{1,3}})|0){_PHONE_SEP}\d{{1,3}}{_PHONE_SEP}\d{{2,4}}{_PHONE_SEP}\d{{2,4}}(?![\w.])"
)

# Football (FootballUnder.txt, FootballOther.txt)
_FOOTBALL_U_RE = re.compile(r"\bU\.?(\d{2})\b")
_FOOTBALL_SCORE_RE = re.compile(
    r"\b(tỉ số|tỷ số|đội hình|thắng|hòa|hoà)(\s+)(\d{1,2}(?:\s?[-|]\s?\d{1,2}){1,3})\b", re.IGNORECASE)

# Political division (PoliticalDivision.txt)
_POLITICAL = {"kp": "khu phố", "q": "quận", "p": "phường", "h": "huyện",
              "tx": "thị xã", "tp": "thành phố", "x": "xã"}
# No whitespace is allowed around the dot. vinorm permits it, but "720P. Thứ ba"
# is a resolution followed by a new sentence, not "phường Thứ ba"; requiring the
# tight form keeps every real address ("Q.1", "P.5", "Q.Bình Thạnh", "TX.Thuận An")
# and makes the rule idempotent.
# A preceding LETTER rules the abbreviation out; a preceding digit must not, or
# "1080P.Hình" would be blocked on the first pass and fire on the second, once the
# digits have become words. What follows is what decides -- see _political_sub.
_POLITICAL_RE = re.compile(rf"(?<![^\W\d_])(KP|Q|P|H|TX|TP|X)\.(?=[\d{_UPPER_CLS}])")

# Roman numerals (RomanNumber.txt)
_ROMAN_CTX = r"(thứ|lần|kỷ|kỉ|kì|kỳ|khóa|khoá|quý|chương|phần|mục|hạng|đợt)"
# Two characters minimum: "hạng C", "mục C" and "hạng X" are grade and section
# labels, not 100 and 10. "(?![A-Za-z])" rather than "\b" so an adjacent digit cannot
# block the match on pass one and then fire on pass two ("thứ III4").
_ROMAN_RE = re.compile(rf"\b{_ROMAN_CTX}\s+([IVXLC]{{2,6}})(?![A-Za-z])")


def _to_roman(n: int) -> str:
    out = ""
    for v, sym in ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
                   (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out += sym
            n -= v
    return out

# Street / Office codes (Street.txt, Office.txt)
_STREET_CTX = (r"(đường|số nhà|số|nhà|địa chỉ|tọa lạc|toạ lạc|xã|thôn|ấp|khu phố|"
               r"căn hộ|cư xá|lô|ngõ|ngách|hẻm|phòng|lớp|đơn vị|quốc lộ|tỉnh lộ)")
_STREET_CODE_RE = re.compile(
    rf"\b{_STREET_CTX}\s+((?=[A-Za-zÀ-ỹ0-9]*\d)(?=[A-Za-zÀ-ỹ0-9]*[A-Za-zÀ-ỹ])[A-Za-zÀ-ỹ0-9]{{2,6}})\b",
    re.IGNORECASE)
_STREET_FRACTION_RE = re.compile(rf"\b{_STREET_CTX}\s+(\d{{1,3}})/(\d{{1,3}})\b", re.IGNORECASE)

# Identifier codes (vinorm Codenumber.txt, context-gated like Street/Office).
# "Mã BHYT: 4413219578" is an identifier, not four billion. Two triggers:
#  * an explicit code word ("mã", "số thẻ", "CCCD", "STK", "IMEI", ...);
#  * any UNGROUPED run of 7+ digits that does not end in three zeros -- a Vietnamese
#    price is dot-grouped ("4.413.219.578") or round ("2000000"), an id is neither.
_ID_TAIL = r"(\s*:?)\s*((?=[A-Za-zÀ-ỹ0-9.-]*\d)[A-Za-zÀ-ỹ0-9]+(?:[.-][A-Za-zÀ-ỹ0-9]+)*)\b"
# The Vietnamese cues keep their diacritics, so the word "ma" (ghost) is not a cue.
_ID_CTX_VI = (r"(m[ã] s[ố]|m[ã]|s[ố] th[ẻ]|s[ố] t[à]i kho[ả]n|s[ố] hi[ệ]u|s[ố] seri|"
              r"serial|s[ố] h[ồ] s[ơ]|bi[ể]n s[ố]|h[ộ] chi[ế]u)")
_ID_CODE_VI_RE = re.compile(rf"\b{_ID_CTX_VI}{_ID_TAIL}", re.IGNORECASE)
# Case-SENSITIVE: under IGNORECASE "PIN" also matched "pin", the Vietnamese word for
# battery and the commonest noun in phone-review text, so "Pin 5000mAh" was read
# digit by digit as if it were a PIN code.
_ID_CTX_UP = r"(CMND|CCCD|BHYT|BHXH|MST|STK|IMEI|PIN|OTP)"
_ID_CODE_UP_RE = re.compile(rf"\b{_ID_CTX_UP}{_ID_TAIL}")
_LONG_DIGIT_RE = re.compile(r"(?<![\d.,])(\d{7,})(?![\d.,])")

# Measurement (Measurement.txt / Measurement_1.txt)
_NUM = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?"
_MEASURE_CUR_RE = re.compile(rf"({_NUM})\s?([%$€£¥₩₭₱฿Ω])")
_CUR_MEASURE_RE = re.compile(rf"([$€£¥₩₭₱฿Ω])\s?({_NUM})")
# µ (MICRO SIGN) and μ (GREEK MU) sit below the À-ỹ range, so list them explicitly
# or "10 µm" never reaches the unit reader.
_UNIT_LETTER = r"[A-Za-zÀ-ỹ\u00b5\u03bcΩ]"
# An exponent digit is only an exponent when the token ends there: "25m2" is square
# metres, but in "1m20" and "2m3cm" the digit is a quantity of its own -- a height of
# one metre twenty, two metres three centimetres -- so nothing may follow it.
_UNIT_EXP = r"(?:[0-9](?![\dA-Za-zÀ-ỹ]))?"
_UNIT_TOKEN_RE = re.compile(
    rf"(?<=\d)\s?(°[A-Za-z]?|{_UNIT_LETTER}{_UNIT_LETTER}*{_UNIT_EXP}"
    rf"(?:/{_UNIT_LETTER}+{_UNIT_EXP})?)"
)

# Numbers (NormalNumber.txt)
_NUMBER_RE = re.compile(
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?(?!\.?\d)"   # 1.234.567 / 1.234,5 -- the lookahead
                                           # keeps "192.168.1.1" out of the grouping
                                           # branch so it reads as a dotted chain
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"      # 1,234,567 / 1,234.5  (US grouping)
    r"|\d+(?:\.\d+)+"                      # 3.1 / 1.2.3
    r"|\d+,\d+"                            # 3,5
    r"|\d+"
)
# A minus sign immediately before a digit is read "âm" -- without this the sign is
# dropped silently and "-3.5" (a DEXA T-score) becomes "3.5". A hyphen BETWEEN two
# numbers is a range, not a negative ("15%-100%", "2 -3 phút"), so the handler also
# looks back past one space; the lookbehind alone cannot see that far.
# A hyphen between two numbers is a range, and Vietnamese says "đến". Left unread the
# relation disappears: 250 of this project's 26,940 sentences carry one ("15-30 Gram",
# "96-98%", "12-18 tháng"). Scores, dates, phones and plates are claimed by their own
# rules earlier, and a third number blocks the match so "4-4-2" is never touched.
_RANGE_RE = re.compile(
    r"(?<![\d.,:/\-\u2013])(\d{1,4}(?:[.,]\d{1,3})?%?)\s*[-\u2013\u2014]\s*"
    r"(\d{1,4}(?:[.,]\d{1,3})?%?)(?![\d.,:/\-\u2013])")


def _range_sub(m: re.Match) -> str:
    before = m.string[max(0, m.start() - 30): m.start()].lower()
    if "trừ" in before or "=" in before or "=" in m.string[m.end(): m.end() + 12]:
        return m.group(0)          # transcribed arithmetic, not a range
    return f"{m.group(1)} đến {m.group(2)}"


_NEG_RE = re.compile(r"(?<![\w%)\]])-(?=\d)")

_DIGIT_LETTER_GLUE = re.compile(r"(?<=\d)(?=[A-Za-zÀ-ỹ])|(?<=[A-Za-zÀ-ỹ])(?=\d)")
_ACRONYM_TOKEN_RE = re.compile(rf"{_VI_WORD}+", re.UNICODE)
_CODE_TOKEN_RE = re.compile(r"\b(?=[A-Za-zÀ-ỹ]*\d)[A-Za-zÀ-ỹ0-9]{2,}\b")
_KEEP_RE = re.compile(r"zzkeepzz(\d+)zzend")

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_ORDINAL = {1: "nhất", 2: "hai", 3: "ba", 4: "tư", 5: "năm"}


# --------------------------------------------------------------------------- numbers
def _read_three(n: int, full: bool) -> str:
    """Read a 0..999 group. ``full`` forces reading leading zeros ("không trăm ...")."""
    hundreds, rest = divmod(n, 100)
    tens, units = divmod(rest, 10)
    words = []
    if hundreds or full:
        words += [_DIGITS[hundreds], "trăm"]
    if tens == 0:
        if units:
            if hundreds or full:
                words.append("lẻ")
            words.append(_DIGITS[units])
    elif tens == 1:
        words.append("mười")
        if units == 5:
            words.append("lăm")
        elif units:
            words.append(_DIGITS[units])
    else:
        words += [_DIGITS[tens], "mươi"]
        if units == 1:
            words.append("mốt")
        elif units == 4:
            words.append("tư")
        elif units == 5:
            words.append("lăm")
        elif units:
            words.append(_DIGITS[units])
    return " ".join(words)


def number_to_vietnamese(n: int) -> str:
    if n == 0:
        return "không"
    if n < 0:
        return "âm " + number_to_vietnamese(-n)
    # Scales reach "triệu tỷ" (10^18). Longer integers fall back to digit-by-digit
    # so a pasted account number can never crash the normalizer.
    if n >= 10**18:
        return read_digits(str(n))
    groups = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000
    scales = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ"]
    parts = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        text = _read_three(g, i != len(groups) - 1)
        if scales[i]:
            text += " " + scales[i]
        parts.append(text)
    return " ".join(parts)


def read_digits(s: str) -> str:
    """Read every digit as a standalone word ("0912" -> "không chín một hai")."""
    return " ".join(_DIGITS[int(d)] for d in s if d.isdigit())


def _ordinal(n: int) -> str:
    return _ORDINAL.get(n) or number_to_vietnamese(n)


def roman_to_int(s: str) -> int | None:
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMAN_VALUES.get(ch)
        if v is None:
            return None
        total += v if v >= prev else -v
        prev = max(prev, v)
    return total or None


def _strip_group(raw: str) -> str:
    """Normalise a written number to plain digits plus an optional ``.`` decimal point."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", raw):      # 1.234.567,89
        raw = raw.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):    # 1,234,567.89
        raw = raw.replace(",", "")
    elif "," in raw:                                             # 3,5
        raw = raw.replace(",", ".")
    return raw


# Python refuses int() on more than 4300 digits, and nobody says a number that long.
# Past 18 digits the run is read digit by digit instead of crashing.
_MAX_INT_DIGITS = 18


def _say_number(raw: str, *, version: bool = False) -> str:
    """Read a written number: Vietnamese or US grouping, integer, decimal or version.

    ``version`` picks "chấm" over "phẩy" for the decimal separator. A lone dot is a
    decimal point far more often than a version dot in this project's corpora (221
    plain numbers against 19 introduced by a version word), so the caller decides
    from the preceding word and "phẩy" is the default.
    """
    s = _strip_group(raw)
    if len(s.replace(".", "")) > _MAX_INT_DIGITS:
        return read_digits(s)
    # "007" is a room or a jersey number, not seven: int() drops the zeros silently.
    if len(s) > 1 and s[0] == "0" and "." not in s:
        return read_digits(s)
    if s.count(".") > 1:                       # version chain 1.2.3
        return " chấm ".join(number_to_vietnamese(int(p)) for p in s.split(".") if p)
    if "." in s:
        int_part, frac = s.split(".", 1)
        head = number_to_vietnamese(int(int_part)) if int_part else ""
        sep = "chấm" if version else "phẩy"
        return f"{head} {sep} {read_digits(frac)}".strip()
    return number_to_vietnamese(int(s))


# Words that introduce a software version or a model number, where the dot is read
# "chấm": "iOS 17.5", "VOS 3.0", "MIUI 12.5", "Nokia 5.3", "bản 2.0".
_VERSION_LEAD = re.compile(
    r"(?i)\b(?:ios|ipados|macos|android|windows|usb|hdmi|bluetooth|wifi|wi-fi|miui|emui|"
    r"oneui|vos|coloros|funtouch|harmonyos|oxygenos|flyme|qc|vooc|dash|nokia|galaxy|iphone|"
    r"ipad|chrome|firefox|java|python|php|unity|opengl|directx|version|ver|bản|v)\s*$")


def _frac_words(n: int, d: int) -> str:
    """A small fraction is "phần" ("ba phần tư"); a ratio or an out-of-n is "trên"
    ("bảy trên chín người", "180 trên 90 mmHg")."""
    if n < d <= 4:
        return f"{number_to_vietnamese(n)} phần {'tư' if d == 4 else number_to_vietnamese(d)}"
    return f"{number_to_vietnamese(n)} trên {number_to_vietnamese(d)}"


# "½ kg" has to keep a digit until step 11, or the unit reader never sees one and the
# kilogram is left unspoken. The character is folded to "1/2" up front and read below.
_VULGAR_ASCII = {ch: f" {unicodedata.normalize('NFKD', ch).replace(chr(0x2044), '/')} "
                 for ch in _FRACTION_CHARS
                 if len(unicodedata.normalize("NFKD", ch).split(chr(0x2044))) == 2
                 and all(p.isdigit() for p in unicodedata.normalize("NFKD", ch).split(chr(0x2044)))}
_VULGAR_RE = re.compile(f"[{''.join(_VULGAR_ASCII)}]")


def _vulgar_sub(m: re.Match) -> str:
    prev = m.string[m.start() - 1] if m.start() else ""
    joiner = " và " if prev.isdigit() else " "      # "2½ kg" is two and a half kilos
    return joiner + _VULGAR_ASCII[m.group(0)].strip() + " "
_SMALL_FRACTION_RE = re.compile(r"(?<![\d/.,])([1-9])\s*/\s*([2-9])(?![\d/.,])")


def _replace_number(match: re.Match) -> str:
    # A 24-character window, not the whole prefix: re-slicing and re-searching every
    # preceding character on every match made a 100k-character paste take ~57 s.
    lead = match.string[max(0, match.start() - 24): match.start()]
    return _say_number(match.group(0), version=bool(_VERSION_LEAD.search(lead)))


# --------------------------------------------------------------------------- units
# Units whose key is also an ordinary Vietnamese word but which really are units
# after a number: "5 ha đất", "5 min". Everything else in the Popular list is a word.
_UNIT_WORD_OK = {"ha", "min"}


def _unit_reading(key: str, *, allow_prefix: bool = True) -> str | None:
    key = key.replace("\u00b5", "\u03bc")   # MICRO SIGN -> GREEK MU, as the prefix table is keyed
    if key in _UNIT_MAP:
        return _UNIT_MAP[key]
    # SI prefix + base unit ("µm" -> "mi cờ rô mét"). Guarded: without the length and
    # case checks "DHA" composed into "đề xi héc ta" (d + ha).
    if allow_prefix and 1 < len(key) <= 4 and key[0] in _PREFIX_UNIT and key[1:] in _UNIT_MAP:
        return f"{_PREFIX_UNIT[key[0]]} {_UNIT_MAP[key[1:]]}"
    return None


def _unit_sub(match: re.Match) -> str:
    tok = match.group(1)
    bare = tok.lstrip("°")
    key = tok.lower().lstrip("°")
    # A few single upper-case letters after a digit are not units in Vietnamese tech
    # and finance speech: "5G"/"4G" network generations, "5M"/"5K"/"5B" magnitudes,
    # "6E" a wifi band, "iPhone 5C" a model. Lower case stays a unit ("500g" grams,
    # "5m" metres) and a degree sign forces the unit reading.
    if not tok.startswith("°") and len(bare) == 1 and bare in _NON_UNIT_UPPER:
        return " " + tok
    # SI prefixes are case sensitive: "dm" is a decimetre, "DM" is not. An all-upper
    # token is an acronym, so only the exact tables may answer for it.
    allow_prefix = tok == tok.lower()
    # A lower-case token that is a real Vietnamese word is a word, not a unit. Without
    # this "3 pha" composed to "ba pi cô héc ta" (p + ha) and "5 ma" to "mi li âm be";
    # 12 words in the Popular list compose that way and 6 more are in the unit table.
    # An upper-case token is an explicit unit or acronym, so "400 IU" still reads.
    if tok == "°":
        return " độ"                   # a bare degree sign still has to be spoken
    if tok == "k":
        return " nghìn"                # "100k" is how Vietnamese writes 100,000
    if bare.rstrip("0123456789") == "tr":
        rest = bare[2:]
        return " triệu" + (f" {_say_number(rest)}" if rest else "")
    before = match.string[max(0, match.start() - 18): match.start()]
    if key == "g" and _NETWORK_BEFORE_RE.search(before):
        return " " + tok               # "mạng 4g", not four grams
    # A lower-case token that is a real Vietnamese word is a word, not a unit. Without
    # this "3 pha" composed to "ba pi cô héc ta" (p + ha) and "5 ma" to "mi li âm be";
    # 12 words in the Popular list compose that way and 6 more are in the unit table.
    # An upper-case token is an explicit unit or acronym, so "400 IU" still reads.
    if allow_prefix and key in _POPULAR and key not in _UNIT_WORD_OK:
        return " " + tok
    reading = _unit_reading(key, allow_prefix=allow_prefix)
    if reading:
        return " " + reading
    if "/" in bare:
        # A side that has no reading is echoed back AS WRITTEN. Echoing the lower-cased
        # lookup key instead turned "H/Pạ1" into "pạ1", which a second pass then took
        # for chat shorthand and rewrote again.
        raw_parts = bare.split("/")
        readings = [_unit_reading(part.lower(), allow_prefix=allow_prefix)
                    for part in raw_parts]
        if any(readings):
            return " " + " trên ".join(r or raw for r, raw in zip(readings, raw_parts))
    return " " + tok


# --------------------------------------------------------------------------- rule subs
def _date_full_sub(m: re.Match, text: str) -> str:
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not 1 <= mo <= 12:
        if 1 <= d <= 12 and 1 <= mo <= 31:
            d, mo = mo, d          # "12/25/2020" is US order
        else:
            return m.group(0)      # "1/0/2020", "1/19/2020": not a date
    if not 1 <= d <= 31:
        return m.group(0)
    before = text[max(0, m.start() - 24): m.start()].rstrip().lower()
    prefix = "" if before.endswith(("ngày", "mùng", "mồng")) else "ngày "
    return f"{prefix}{number_to_vietnamese(d)} tháng {number_to_vietnamese(mo)} năm {number_to_vietnamese(y)}"


def _date_dm_sub(m: re.Match) -> str:
    d, mo = int(m.group(2)), int(m.group(3))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    return f"{m.group(1)} {number_to_vietnamese(d)} tháng {number_to_vietnamese(mo)}"


def _date_range_dd_sub(m: re.Match) -> str:
    ctx, d1, d2, mo = m.group(1), *(int(m.group(i)) for i in (2, 3, 4))
    return (f"{ctx} {number_to_vietnamese(d1)} đến {number_to_vietnamese(d2)} "
            f"tháng {number_to_vietnamese(mo)}")


def _date_range_dmdm_sub(m: re.Match) -> str:
    ctx, d1, m1, d2, m2 = m.group(1), *(int(m.group(i)) for i in (2, 3, 4, 5))
    return (f"{ctx} {number_to_vietnamese(d1)} tháng {number_to_vietnamese(m1)} đến "
            f"{number_to_vietnamese(d2)} tháng {number_to_vietnamese(m2)}")


def _month_year_sub(m: re.Match) -> str:
    mo = int(m.group(1))
    if not 1 <= mo <= 12:
        return m.group(0)          # "tháng 19/2026" is not a month
    return f"tháng {number_to_vietnamese(mo)} năm {number_to_vietnamese(int(m.group(2)))}"


def _month_range_sub(m: re.Match) -> str:
    ctx, m1, m2, y = m.group(1), *(int(m.group(i)) for i in (2, 3, 4))
    head = f"{ctx} tháng" if ctx.lower() == "từ" else ctx     # "từ tháng 1", "tháng 1"
    return (f"{head} {number_to_vietnamese(m1)} đến tháng {number_to_vietnamese(m2)} "
            f"năm {number_to_vietnamese(y)}")


def _time_hm_sub(m: re.Match) -> str:
    h, mi, ampm = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
    out = f"{number_to_vietnamese(h)} giờ {number_to_vietnamese(mi)} phút"
    if ampm == "am":
        out += " sáng"
    elif ampm == "pm":
        out += " chiều"
    return f" {out} "


def _time_g_sub(m: re.Match) -> str:
    before = m.string[max(0, m.start() - 14): m.start()]
    after = m.string[m.end(): m.end() + 10]
    if _CLOCK_BEFORE_RE.search(before) or _CLOCK_AFTER_RE.match(after):
        return f" {number_to_vietnamese(int(m.group(1)))} giờ "
    return m.group(0)                  # leave it: the unit reader says "gam"


def _time_ampm_sub(m: re.Match) -> str:
    part = "sáng" if m.group(2).lower() == "a" else "chiều"
    return f" {number_to_vietnamese(int(m.group(1)))} giờ {part} "


def _time_h_sub(m: re.Match) -> str:
    return f" {number_to_vietnamese(int(m.group(1)))} giờ "


def _phone_sub(m: re.Match) -> str:
    """Read a phone number digit by digit, keeping any parenthesis pair balanced."""
    # The separator class can swallow the closing paren of an area code while the
    # opening one stays outside the match; re-emit the surplus so pairs balance.
    raw = m.group(0)
    spoken = read_digits(raw)
    surplus = raw.count(")") - raw.count("(")
    if surplus > 0:
        spoken += ")" * surplus
    return spoken


def _football_u_sub(m: re.Match) -> str:
    return f"u {number_to_vietnamese(int(m.group(1)))}"


def _football_score_sub(m: re.Match) -> str:
    scores = re.findall(r"\d{1,2}", m.group(3))
    return m.group(1) + m.group(2) + " ".join(number_to_vietnamese(int(s)) for s in scores)


# What follows a real division abbreviation is a number ("Q.1") or a place name,
# either all caps ("TP.HCM") or two capitalised words ("H.Bình Chánh"). An anonymised
# initial in press copy is followed by an ordinary sentence ("Nguyễn Văn H.Đã bị
# bắt"), which this declines.
_PLACE_AFTER_RE = re.compile(
    # A slash counts as a separator between the words of a place name only so that
    # this test gives the same answer before and after step 12c removes the slash.
    rf"^(?:\d|[{_UPPER_CLS}]{{2,}}|[{_UPPER_CLS}][{_LOWER_CLS}]+(?:[\s/]+[{_UPPER_CLS}][{_LOWER_CLS}]+)+)")


def _political_sub(m: re.Match) -> str:
    if not _PLACE_AFTER_RE.match(m.string[m.end():]):
        return m.group(0)
    return _POLITICAL[m.group(1).lower()] + " "


def _roman_sub(m: re.Match) -> str:
    ctx, roman = m.group(1), m.group(2)
    n = roman_to_int(roman)
    if n is None or n > 400 or _to_roman(n) != roman.upper():
        return m.group(0)          # "IIII" and "VV" are not numerals
    word = _ordinal(n) if ctx.lower() in ("thứ", "hạng") else number_to_vietnamese(n)
    return f"{ctx} {word}"


def _street_code_sub(m: re.Match) -> str:
    code = m.group(2)
    # A measurement is not a code: "phòng 25m2" is an area, "phòng 12B" a room. If the
    # non-numeric tail is a known unit, leave it for the unit reader.
    tail = code.lstrip("0123456789")
    if tail and len(tail) > 1 and _unit_reading(tail.lower()):
        return m.group(0)
    spoken = [_DIGITS[int(c)] if c.isdigit() else _LETTER_VN.get(c.lower(), c) for c in code]
    return f"{m.group(1)} " + " ".join(spoken)


def _street_fraction_sub(m: re.Match) -> str:
    ctx, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    joint = "tháng" if 1 <= b <= 12 else "trên"
    return f"{ctx} {number_to_vietnamese(a)} {joint} {number_to_vietnamese(b)}"


def _currency_word(sym: str) -> str:
    """Currency and symbol tables are keyed in lower case ("ω"), input may be "Ω"."""
    return _CURRENCY.get(sym) or _CURRENCY.get(sym.lower()) or _SYMBOLS.get(sym.lower()) or sym


def _spell_code(code: str) -> str:
    """Read an identifier character by character, keeping separators audible."""
    out = []
    for ch in code:
        if ch.isdigit():
            out.append(_DIGITS[int(ch)])
        elif ch.isalpha():
            out.append(_LETTER_VN.get(ch.lower(), ch))
    return " ".join(out)


def _id_code_sub(m: re.Match) -> str:
    code = m.group(3)
    # defer to the unit reader when the tail is a unit ("mã 25m2" is not an id)
    tail = code.lstrip("0123456789")
    if tail and len(tail) > 1 and _unit_reading(tail.lower()):
        return m.group(0)
    return f"{m.group(1)}{m.group(2)} {_spell_code(code)}"


# A classifier after the run, or a money/population cue before it, marks a quantity:
# "dân số 98168833 người" is a count, not an identifier.
_QUANTITY_AFTER_RE = re.compile(
    r"(?i)^\s*(người|d[âa]n|đ[ồo]ng|usd|vnd|đ|l[ưu][ợo]t|c[áa]i|con|chi[ếe]c|t[ấa]n|kg|"
    r"ki l[ôo]|m[ée]t|ha|ng[àa]y|n[ăa]m|gi[ờo]|l[ầa]n|t[ỷy]|tri[ệe]u|ngh[ìi]n|c[ổo] phi[ếe]u)\b")
_QUANTITY_BEFORE_RE = re.compile(
    r"(?i)(gi[áa]|d[âa]n s[ốo]|doanh thu|t[ổo]ng|kho[ảa]ng|h[ơo]n|g[ầa]n|đ[ạa]t|l[êe]n t[ớo]i|"
    r"thu|chi|l[ãa]i|l[ỗo]|v[ốo]n|m[ấa]t|c[òo]n|b[áa]n|mua|tr[ịi] gi[áa])\s*$")


def _long_digit_sub(m: re.Match) -> str:
    run = m.group(1)
    # A round number is a quantity, not an id: "2000000" stays "hai triệu".
    if run.endswith("000"):
        return run
    if (_QUANTITY_AFTER_RE.match(m.string[m.end():])
            or _QUANTITY_BEFORE_RE.search(m.string[max(0, m.start() - 20): m.start()])):
        return run
    return " " + _spell_code(run) + " "


def _measure_cur_sub(m: re.Match) -> str:
    """number + symbol (vinorm Measurement_1). Padded: "5%pin" must not weld."""
    return f" {_say_number(m.group(1))} {_currency_word(m.group(2))} "


def _cur_measure_sub(m: re.Match) -> str:
    """symbol + number: "$5" is said "năm đô la", not "đô la năm"."""
    return f" {_say_number(m.group(2))} {_currency_word(m.group(1))} "


# --------------------------------------------------------------------------- acronyms
def _expand_acronym(word: str) -> str:
    key = word.lower()
    if key in _ACRONYM_SKIP:
        return word
    reading = _ACRONYM_OVERRIDE.get(key) or _ACRONYMS.get(key)
    if reading is None:
        return word
    if key in _POPULAR:            # a real Vietnamese word is never an acronym
        return word
    if word.isupper() and (len(word) >= 4 or key in _ACRONYM_SHORT_ALLOW):
        return reading
    if word == key and key in _ACRONYM_LOWER_ALLOW:
        return reading
    # Lower-case acronyms are typed informally ("so bhyt", "cmnd", "tphcm", "ubnd").
    # Measured on the table itself: of the 2,932 keys of four or more characters that
    # are not Vietnamese words, exactly two are also English words (blocked below),
    # so expanding the rest cannot damage an English span in code-switched text.
    if len(word) >= 4 and key not in _ACRONYM_EN_BLOCK:
        return reading
    return word


def spell_letters(word: str, english: bool = False) -> str:
    """Read a token letter by letter ("VTV" -> "vê tê vê"), vinorm's LetterSound tables."""
    table = _LETTER_EN if english else _LETTER_VN
    return " ".join(
        _DIGITS[int(c)] if c.isdigit() else table.get(c.lower(), c)
        for c in word if c.isalnum()
    )


def _is_spellable(word: str) -> bool:
    """An all-caps token with no vowel can never be a Vietnamese word, so spell it."""
    if not word.isupper() or len(word) < 2 or not word.isalpha():
        return False
    if word.lower() in _POPULAR or word.lower() in _ACRONYMS:
        return False
    return not re.search(r"[AEIOUYĂÂÊÔƠƯ]", word)


def _to_alpha(digits: str) -> str:
    """Map an index to letters so a placeholder contains no digit."""
    return "".join("abcdefghij"[int(d)] for d in digits)


# --------------------------------------------------------------------------- main
def normalize_vietnamese(
    text: str,
    *,
    teencode: bool = True,
    spell_acronyms: bool = False,
    read_codes: bool = False,
) -> str:
    """Normalize Vietnamese text for TTS.

    Args:
        text: raw input.
        teencode: expand vinorm's 484-entry chat-shorthand table ("ko" -> "không").
        spell_acronyms: read unknown all-caps vowel-less tokens letter by letter
            ("VTV" -> "vê tê vê"). Off by default: it destroys the English spans the
            code-switching benchmark scores against.
        read_codes: read every remaining digit+letter token character by character
            (vinorm's Codenumber catch-all). Off by default: it would turn
            "iPhone 15" into "iPhone một năm".
    """
    if not text:
        return text
    # Decomposed Vietnamese reaches us from real sources (3 % of UIT-ViSFD): "ế" as
    # "e" + combining acute defeats every diacritic pattern here and costs the
    # tokenizer ~20 % more tokens. Drop the invisible characters FIRST: a zero-width
    # space sitting between a base letter and its combining mark blocks composition,
    # and stripping it afterwards would leave decomposed output behind.
    text = _ZW_SEP_RE.sub(" ", text)
    text = _LINE_SEP_RE.sub(" ", text)
    text = _CONTROL_RE.sub("", text)
    text = _DROP_RE.sub("", text)
    text = text.translate(_LOOKALIKE)
    text = _SUPERSCRIPT_RE.sub(_superscript_sub, text)
    text = _VULGAR_RE.sub(_vulgar_sub, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    # Tidy "v.v ." -> "v.v." up front, otherwise the end-of-pipeline cleanup creates
    # patterns the rules below would have matched (the function was not idempotent).
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)

    # 1. protect emails and URLs from every later rule
    stash: list[str] = []

    def _stash(m: re.Match) -> str:
        stash.append(m.group(0))
        return f" zzkeepzz{len(stash) - 1}zzend "

    text = _EMAIL_RE.sub(_stash, text)
    text = _URL_RE.sub(_stash, text)
    text = _KEEP_RE.sub(lambda m: f"zzkeepzz{_to_alpha(m.group(1))}zzend", text)

    # 2. chat shorthand, on the user's words only. Running this after the numeric
    #    rules re-mapped their output ("₩" -> "won" -> teencode -> "quan").
    if teencode and _TEENCODE:
        def _teen(m: re.Match) -> str:
            w = m.group(0)
            # Chat shorthand is written lower case. Applying it to upper case turned
            # the ingredient "UC-II" into "úc-II" ("uc" -> "úc" in the table).
            if w != w.lower() or w in _POPULAR:
                return w
            return _TEENCODE.get(w) or w

        text = _ACRONYM_TOKEN_RE.sub(_teen, text)

    # 3. dates, most specific first
    text = _DATE_RANGE_DMDM_RE.sub(_date_range_dmdm_sub, text)
    text = _DATE_RANGE_DD_RE.sub(_date_range_dd_sub, text)
    text = _DATE_RANGE_BARE_RE.sub(_date_range_bare_sub, text)
    text = _MONTH_RANGE_RE.sub(_month_range_sub, text)
    text = _MONTH_YEAR_RE.sub(_month_year_sub, text)
    _buf = text
    text = _DATE_FULL_RE.sub(lambda m: _date_full_sub(m, _buf), text)
    text = _DATE_DM_RE.sub(_date_dm_sub, text)

    # 3. clock times
    text = _TIME_HM_RE.sub(_time_hm_sub, text)
    text = _TIME_H_RE.sub(_time_h_sub, text)
    text = _TIME_G_RE.sub(_time_g_sub, text)
    text = _TIME_AMPM_RE.sub(_time_ampm_sub, text)

    # 4. phone numbers and hotlines, read digit by digit
    text = _HOTLINE_RE.sub(lambda m: read_digits(m.group(0)), text)
    text = _PHONE_RE.sub(_phone_sub, text)

    # 5. football scores and age groups
    text = _FOOTBALL_SCORE_RE.sub(_football_score_sub, text)
    text = _FOOTBALL_U_RE.sub(_football_u_sub, text)

    # 6. administrative abbreviations ("Q.1" -> "quận 1"). The compound place
    #    names run first: the generic rule would reduce "TP.HCM" to "thành phố HCM".
    text = re.sub(r"\bTP\.?\s*HCM\b", "thành phố Hồ Chí Minh", text)
    text = _POLITICAL_RE.sub(_political_sub, text)

    # 7. roman numerals in an ordinal context
    text = _ROMAN_RE.sub(_roman_sub, text)

    # 8. identifier codes first: the address list below contains a bare "số", which
    #    would otherwise claim the "số" inside "biển số" and leave the plate half-read
    text = _ID_CODE_UP_RE.sub(_id_code_sub, text)
    text = _ID_CODE_VI_RE.sub(_id_code_sub, text)
    text = _LONG_DIGIT_RE.sub(_long_digit_sub, text)

    # 9. street / office codes, read character by character
    text = _STREET_FRACTION_RE.sub(_street_fraction_sub, text)
    text = _STREET_CODE_RE.sub(_street_code_sub, text)

    # 9b. numeric ranges and signs, before the percent and currency readers turn the
    #     sides into words: "-12%" lost its sign entirely when the negative rule ran
    #     after the percent reader had already consumed the digits.
    text = _RANGE_RE.sub(_range_sub, text)

    def _neg_sub(m: re.Match) -> str:
        before = text[max(0, m.start() - 24): m.start()].rstrip()
        if before and (before[-1].isdigit() or before[-1] in "%)"):
            return m.group(0)          # a range, not a negative
        return " âm "

    text = _NEG_RE.sub(_neg_sub, text)

    # 10. currency and percent, either order
    text = _MEASURE_CUR_RE.sub(_measure_cur_sub, text)
    text = _CUR_MEASURE_RE.sub(_cur_measure_sub, text)

    # 11. units glued to the number ("25m2", "31°C", "100 km/h") before the
    #     digit-letter split would tear "25m2" into "25 m 2"
    text = _UNIT_TOKEN_RE.sub(_unit_sub, text)

    # 12. "10a" -> "10 a" so number words cannot weld to a letter ("mườia")
    text = _DIGIT_LETTER_GLUE.sub(" ", text)

    # 12b. single-digit fractions, after the unit reader has had the digits
    text = _SMALL_FRACTION_RE.sub(lambda m: f" {_frac_words(int(m.group(1)), int(m.group(2)))} ", text)

    # 12c. "và/hoặc" is not a ratio. This must run while the numbers are still
    #      digits: after step 14 both sides of "180/90" are words and the rule
    #      would eat the slash that carries the ratio.
    text = re.sub(rf"(?<=[{_LOWER_CLS}{_UPPER_CLS}])\s*/\s*(?=[{_LOWER_CLS}{_UPPER_CLS}])",
                  " ", text)

    # 13. fixed abbreviations
    for pat, rep in ABBREVIATIONS.items():
        text = re.sub(pat, rep, text)

    # 14. remaining numbers
    if read_codes:
        text = _CODE_TOKEN_RE.sub(lambda m: spell_letters(m.group(0)), text)
    text = _NUMBER_RE.sub(_replace_number, text)

    # 15. chat shorthand, then acronyms (conservative), then optional letter spelling
    def _token_sub(m: re.Match) -> str:
        w = m.group(0)
        e = _expand_acronym(w)
        if e != w:
            return e
        if spell_acronyms and _is_spellable(w):
            return spell_letters(w)
        return w

    text = _ACRONYM_TOKEN_RE.sub(_token_sub, text)

    # 16. standalone symbols, padded so the reading cannot weld to a letter
    #     ("Rh+" -> "Rh cộng", not "Rhcộng")
    if _SYMBOLS:
        def _sym_sub(m: re.Match) -> str:
            ch = m.group(0)
            # The currency signs live in their own table; without this fallback a
            # standalone "£" or "€" reached the model unspoken (4 cases in 175k
            # held-out lines, all in symbol spam).
            word = (_SYMBOLS.get(ch) or _SYMBOLS.get(ch.lower())
                    or _CURRENCY.get(ch) or _CURRENCY.get(ch.lower()))
            return f" {word} " if word else ch

        text = re.sub(rf"[@$^&*+=<>≠/±×÷∞©%£¥€₩₭₱฿Ω\u2030\u2103\u2109{_FRACTION_CHARS}]",
                      _sym_sub, text)

    text = text.replace('"', "").replace("“", "").replace("”", "")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    # A rule that drops a base letter can orphan its combining mark onto the previous
    # one, so the output is composed again: NFC in, NFC out, and normalize(normalize(x))
    # == normalize(x). Done before the stash is restored so protected spans stay byte
    # for byte what the user wrote.
    text = unicodedata.normalize("NFC", text)

    # 17. restore protected emails / URLs verbatim
    for i, original in enumerate(stash):
        text = text.replace(f"zzkeepzz{_to_alpha(str(i))}zzend", original)
    return text
