"""Tests for the Vietnamese branch of TextNormalizer.

Run with the project venv:
    Confucius4-TTS/.venv/bin/python -m pytest tests/test_text_normalizer_vi.py -q
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from confuciustts.frontend.text_normalizer import TextNormalizer
from confuciustts.frontend.vi_normalizer import number_to_vietnamese


@pytest.fixture(scope="module")
def norm() -> TextNormalizer:
    return TextNormalizer()


# --------------------------------------------------------------------------- normalize()


class TestNormalizeVi:
    def test_plain_integer(self, norm):
        assert norm.normalize("có 2 ngày", "vi") == "có hai ngày"

    def test_thousands_with_dots(self, norm):
        assert norm.normalize("Giá 1.250.000 đ", "vi") == "Giá một triệu hai trăm năm mươi nghìn đồng"

    def test_decimal_comma(self, norm):
        # Vietnamese law: x1 is "mốt" after "mươi" (ba mươi MỐT, not một)
        assert norm.normalize("nhiệt độ 31,5 độ", "vi") == "nhiệt độ ba mươi mốt phẩy năm độ"

    def test_date(self, norm):
        # the date itself stays as digits in the normalized string; the number
        # regex then reads it out in full — either way no digits survive
        out = norm.normalize("Hôm nay là 22/9/2026", "vi")
        assert "ngày" in out and "tháng" in out and "năm" in out
        assert "/" not in out
        assert "2026" not in out

    def test_percent(self, norm):
        out = norm.normalize("tăng 50% tháng này", "vi")
        assert "năm mươi phần trăm" in out
        assert "%" not in out

    def test_abbreviation(self, norm):
        out = norm.normalize("UBND thành phố", "vi")
        assert "ủy ban nhân dân" in out

    def test_date_no_double_ngày(self, norm):
        # text that already says "ngày" must not get a second one
        out = norm.normalize("ngày 22/9/2026", "vi")
        assert out.count("ngày") == 1
        out = norm.normalize("vào ngày 1/1/2024", "vi")
        assert out.count("ngày") == 1
        # bare dates still get the prefix
        out = norm.normalize("22/9/2026 là chủ nhật", "vi")
        assert out.startswith("ngày")

    def test_dotted_version_chain(self, norm):
        out = norm.normalize("USB 3.0 và iOS 18.2", "vi")
        assert "ba chấm không" in out
        assert "mười tám chấm hai" in out

    def test_no_english_reading(self, norm):
        # "2" must never become "two" (the old behaviour via normalize_english)
        out = norm.normalize("2 con cá", "vi")
        assert "hai" in out
        assert "two" not in out

    def test_auto_detect_vietnamese(self, norm):
        # auto: tone-marked text routes to the vi branch, not the English one
        out = norm.normalize("Hôm nay có 3 người", language="auto")
        assert "ba" in out
        assert "three" not in out

    def test_stacked_tone_marks(self, norm):
        # ế/ệ are tone marks stacked on â/ê — "tiếng Việt" was the reported
        # miss that sent real Vietnamese sentences to the English normalizer
        assert norm.contains_vietnamese("tiếng Việt")
        assert norm.contains_vietnamese("Chiều thứ nhất")

    def test_uppercase_marked_text(self, norm):
        # uppercase marks route via casefold; a bare no-mark word stays en
        assert norm.contains_vietnamese("TIẾNG VIỆT")
        assert norm.contains_vietnamese("HỒ CHÍ MINH")
        assert not norm.contains_vietnamese("MINH")  # no diacritic — ambiguous

    def test_auto_ascii_number_without_marks_stays_english(self, norm):
        # ASCII text is ambiguous; without tone marks it must keep the en path
        assert norm.normalize("I have 2 dogs", language="auto") == "I have two dogs"


# --------------------------------------------------------------------------- domain rules (vinorm-ported)


class TestDomainsVi:
    def test_phone_digit_by_digit(self, norm):
        # a 10-digit run must never be read as a magnitude
        out = norm.normalize("gọi số 0912345678 nhé", "vi")
        assert "chín trăm mười hai triệu" not in out
        assert "không chín một hai ba bốn năm sáu bảy tám" in out

    def test_phone_grouped(self, norm):
        out = norm.normalize("gọi 0912 345 678 nhé", "vi")
        assert "không chín một hai ba bốn năm sáu bảy tám" in out

    def test_hotline(self, norm):
        out = norm.normalize("hotline: 1900 636 839", "vi")
        assert "một chín không không sáu ba sáu tám ba chín" in out

    def test_phone_paren_area_code(self, norm):
        out = norm.normalize("liên hệ (024) 3825-1234", "vi")
        assert out.count("(") == out.count(")")
        assert "không hai bốn" in out

    def test_year_range_not_phone(self, norm):
        # 4-digit years must keep magnitude reading, not digit-by-digit
        out = norm.normalize("từ 1900 tới 2000", "vi")
        assert "một nghìn chín trăm" in out and "hai nghìn" in out

    def test_email_preserved(self, norm):
        out = norm.normalize("gửi tới long.nguyen@gmail.com nhé", "vi")
        assert "long.nguyen@gmail.com" in out
        assert "a còng" not in out

    def test_url_preserved(self, norm):
        out = norm.normalize("xem tại https://example.com.vn/info?id=25", "vi")
        assert "https://example.com.vn/info?id=25" in out
        assert "hai lăm" not in out

    def test_standalone_symbol_at(self, norm):
        # an @ NOT inside an email address still reads "a còng"
        out = norm.normalize("gửi tới @admin", "vi")
        assert "a còng" in out

    def test_unit_after_digit(self, norm):
        assert "độ xê" in norm.normalize("nhiệt độ 31°C", "vi")
        assert "ki lô oát giờ" in norm.normalize("công suất 5 kWh", "vi")
        assert "mi li mét thủy ngân" in norm.normalize("huyết áp 120 mmHg", "vi")
        assert "ki lô mét một giờ" in norm.normalize("tốc độ 100 km/h", "vi")

    def test_unit_glued_area(self, norm):
        # "25m2" must not tear into "hai mươi lăm mét hai"
        assert "mét vuông" in norm.normalize("diện tích 25m2", "vi")
        assert "mét khối" in norm.normalize("thể tích 2m3", "vi")

    def test_street_fraction_reads_month(self, norm):
        # "đường 30/4" is a proper name, not "xuyệt"
        out = norm.normalize("số nhà 12 đường 30/4", "vi")
        assert "xuyệt" not in out
        assert "tháng bốn" in out

    def test_street_fraction_non_month(self, norm):
        # 25/7: "trên" reading would sound like a fraction; months 1-12 always
        # read "tháng" in address names (Ngõ 25/7 ~ the 25/7 alley)
        out = norm.normalize("ngõ 25/7 vào nhà", "vi")
        assert "xuyệt" not in out
        assert "tháng bảy" in out

    def test_symbol_arithmetic(self, norm):
        out = norm.normalize("5 + 3 bằng 8", "vi")
        assert "cộng" in out

    def test_acronym_from_vinorm_dict(self, norm):
        # entries ported from vinorm's Acronyms.txt
        assert "cảnh sát giao thông" in norm.normalize("CSGT xử lý", "vi")
        assert "bảo hiểm xã hội" in norm.normalize("đóng BHXH đầy đủ", "vi")


# --------------------------------------------------------------------------- segment_text


class TestSegmentVi:
    def test_comma_split_long_clauses(self, norm):
        # commas break Vietnamese clauses once clauses exceed max_tokens
        text = "Sáng nay trời đẹp, gió nhẹ, chúng ta đi chơi. Chiều về mưa rơi, đường trơn, mọi người cẩn thận."
        segments = norm.segment_text(
            text, lambda t: t.split(), language="vi", max_tokens=8, min_tokens=0, merge_threshold=0
        )
        assert any("Sáng nay trời đẹp," in s for s in segments)
        assert any("Chiều về mưa rơi," in s for s in segments)

    def test_hard_cut_never_splits_syllable(self, norm):
        # One long sentence without internal punctuation -> word-boundary cut
        text = "Một câu rất dài không hề có dấu chấm nào ở giữa cả vẫn tiếp tục kéo dài hơn nữa"
        segments = norm.segment_text(
            text, lambda t: t.split(), language="vi", max_tokens=6, min_tokens=0, merge_threshold=0
        )
        assert len(segments) > 1
        for seg in segments:
            # no glued syllables: every original syllable appears intact
            assert "Mộtcâu" not in seg
            assert seg.strip().split()  # not empty

    def test_hard_cut_preserves_all_syllables(self, norm):
        text = " ".join(["Hôm nay tôi đi chợ mua rất nhiều thứ"] * 5)
        segments = norm.segment_text(
            text, lambda t: t.split(), language="vi", max_tokens=10, min_tokens=0, merge_threshold=0
        )
        joined = " ".join(s.strip() for s in segments)
        assert joined.split() == text.split()

    def test_zh_path_unchanged(self, norm):
        # Chinese merge behaviour is pre-existing and must not be touched
        segments = norm.segment_text("你好。世界。", lambda t: [t] * len(t), language="zh")
        assert len(segments) == 1  # short segments merge back, zh strips trailing 。

    def test_english_path_unchanged(self, norm):
        # short segments merge into one under max_tokens=80 — stock behaviour
        segments = norm.segment_text("Hello world. Another one.", lambda t: t.split(), language="en")
        assert segments == ["Hello world. Another one."]


# --------------------------------------------------------------------------- unit helpers


class TestNumberToVietnamese:
    @pytest.mark.parametrize("n,expected", [
        (0, "không"),
        (5, "năm"),
        (15, "mười lăm"),
        (21, "hai mươi mốt"),
        (24, "hai mươi tư"),
        (105, "một trăm lẻ năm"),
        (101, "một trăm lẻ một"),
        (1250, "một nghìn hai trăm năm mươi"),
    ])
    def test_numbers(self, n, expected):
        assert number_to_vietnamese(n) == expected


# --------------------------------------------------------------------------- vinorm-port guards


class TestAcronymConservatism:
    """vinorm's table has 2-letter keys that collide with ordinary words, so only
    tokens written as acronyms are expanded (see _expand_acronym)."""

    @pytest.mark.parametrize("text", [
        "liều mg thấp",       # mg = milligram, table says "mật gấu"
        "Sp này rất tốt",     # sp = sản phẩm in reviews, table says "sư phạm"
        "nv tư vấn tốt",      # nv = nhân viên, table says "nhân văn"
        "kh hài lòng",        # kh = khách hàng, table says "khoa học"
        "ch này rẻ",          # ch = cửa hàng, table says "cộng hòa"
    ])
    def test_lowercase_collisions_left_alone(self, text):
        from confuciustts.frontend.vi_normalizer import normalize_vietnamese
        assert normalize_vietnamese(text) == text

    @pytest.mark.parametrize("text,expected_fragment", [
        ("Mua ở TGDĐ", "thế giới di động"),
        ("UBND tỉnh", "ủy ban nhân dân"),
        ("ĐT của tôi", "điện thoại"),
        ("đt pin bền", "điện thoại"),     # allowlisted lower-case form
        ("ntn là đúng", "như thế nào"),
    ])
    def test_real_acronyms_still_expand(self, text, expected_fragment):
        from confuciustts.frontend.vi_normalizer import normalize_vietnamese
        assert expected_fragment in normalize_vietnamese(text)

    def test_units_after_digit_unaffected(self, norm):
        assert "mi li gam" in norm.normalize("Uống 500 mg mỗi ngày", "vi")

    def test_tech_acronyms_not_rewritten(self, norm):
        out = norm.normalize("Máy PC và TB 1 cái", "vi")
        assert "phòng chống" not in out and "thông báo" not in out

    def test_ct_scan_not_expanded(self, norm):
        # "CT" is a CT scan in clinical speech, not "chương trình"
        assert "chương trình" not in norm.normalize("Chụp CT phổi", "vi")

    @pytest.mark.parametrize("text,expected_fragment", [
        ("NV tư vấn nhiệt tình", "nhân viên"),   # not "nhân văn"
        ("SP này đẹp", "sản phẩm"),              # not "sư phạm"
        ("CH bán hàng", "cửa hàng"),             # not "cộng hòa"
        ("KH phàn nàn", "khách hàng"),           # not "khoa học"
    ])
    def test_consumer_sense_overrides(self, norm, text, expected_fragment):
        assert expected_fragment in norm.normalize(text, "vi")


class TestSymbolSpacing:
    """Symbol readings must not weld onto the neighbouring letter."""

    @pytest.mark.parametrize("text,expected", [
        ("Rh+ của thai", "Rh cộng của thai"),
        ("nồng độ 5 ng/ml", "nồng độ năm nanogam trên mi li lít"),
        ("C++ ổn", "C cộng cộng ổn"),
    ])
    def test_spacing(self, text, expected):
        from confuciustts.frontend.vi_normalizer import normalize_vietnamese
        assert normalize_vietnamese(text) == expected

    def test_no_glued_symbol_word(self, norm):
        import re
        out = norm.normalize("Kháng thể Rh+ và CD4+ với tỉ lệ 5 mmol/l", "vi")
        assert not re.search(r"[A-Za-zÀ-ỹ](?:cộng|trên)|(?:cộng|trên)[A-Za-zÀ-ỹ]", out)


class TestCodeSwitching:
    """Vietnamese with English inserted: the English must survive verbatim so the
    benchmark's English-span scoring still works, and digit+letter blobs must not
    be read as units."""

    @pytest.mark.parametrize("text,spans", [
        ("Team đang train model deep learning trên server mới.",
         ["Team", "train", "model", "deep", "learning", "server"]),
        ("Bạn nhớ check mail và update lại file báo cáo trước deadline.",
         ["check", "mail", "update", "file", "deadline"]),
        ("Anh ấy nói rằng the model is not ready for production yet.",
         ["the", "model", "is", "not", "ready", "for", "production", "yet"]),
        ("Dùng Figma, Notion và Slack cho team.", ["Figma", "Notion", "Slack", "team"]),
    ])
    def test_english_spans_survive(self, norm, text, spans):
        import re
        out = norm.normalize(text, "vi")
        for w in spans:
            assert re.search(r"\b" + re.escape(w) + r"\b", out), f"{w!r} lost in {out!r}"

    @pytest.mark.parametrize("text,expected", [
        ("mạng 5G rất nhanh", "mạng năm G rất nhanh"),   # network generation, not "gam"
        ("điện thoại 4G", "điện thoại bốn G"),
        ("màn 4K", "màn bốn K"),
        ("doanh thu 5M đô", "doanh thu năm M đô"),
    ])
    def test_single_uppercase_letter_is_not_a_unit(self, text, expected):
        from confuciustts.frontend.vi_normalizer import normalize_vietnamese
        assert normalize_vietnamese(text) == expected

    @pytest.mark.parametrize("text,fragment", [
        ("500g thịt", "gam"),          # lower case stays a unit
        ("cao 5m", "mét"),
        ("65W sạc nhanh", "oát"),      # unambiguous upper-case unit
        ("31°C hôm nay", "độ xê"),     # degree sign forces the unit reading
    ])
    def test_real_units_still_expand(self, text, fragment):
        from confuciustts.frontend.vi_normalizer import normalize_vietnamese
        assert fragment in normalize_vietnamese(text)

    def test_brand_numbers_read_in_vietnamese(self, norm):
        out = norm.normalize("Mua iPhone 15 Pro Max", "vi")
        assert "iPhone mười lăm Pro Max" in out

    def test_english_acronyms_not_rewritten(self, norm):
        out = norm.normalize("Dùng AI, API và GPU cho ML", "vi")
        for w in ("AI", "API", "GPU", "ML"):
            assert w in out

    @pytest.mark.parametrize("word", ["HCV", "VP", "PTN", "TPN", "NMN", "HVT", "GH", "DNA", "PCR"])
    def test_scientific_abbreviations_kept(self, word):
        # the table reads HCV as "huy chương vàng", VP as "văn phòng" ...
        from confuciustts.frontend.vi_normalizer import normalize_vietnamese
        assert word in normalize_vietnamese(f"Xét nghiệm {word} dương tính")

    @pytest.mark.parametrize("text,fragment", [
        ("Mua ở TGDĐ", "thế giới di động"),      # 4+ letters: still expands
        ("UBND tỉnh", "ủy ban nhân dân"),
        ("CSKH trả lời nhanh", "chăm sóc khách hàng"),
        ("NV ở CH", "nhân viên"),                 # short, but allowlisted
        ("VN và TP", "Việt Nam"),   # ABBREVIATIONS handles VN before the acronym stage
    ])
    def test_wanted_acronyms_survive_the_length_rule(self, norm, text, fragment):
        assert fragment in norm.normalize(text, "vi")


class TestUnicodeRobustness:
    """Decomposed Vietnamese reaches us from real sources (3 % of UIT-ViSFD), and
    invisible characters must never reach the model."""

    @pytest.mark.parametrize("text", [
        "tiếng Việt", "Chiều thứ nhất", "trí tuệ nhân tạo", "công nghệ mới", "TIẾNG VIỆT",
    ])
    def test_detection_handles_nfd(self, norm, text):
        import unicodedata
        assert norm.contains_vietnamese(unicodedata.normalize("NFD", text))

    def test_nfd_input_normalizes_like_nfc(self, norm):
        import unicodedata
        s = "Hôm nay trời đẹp, Nguyễn Văn Đức nghiên cứu trí tuệ nhân tạo."
        assert norm.normalize(unicodedata.normalize("NFD", s), "vi") == norm.normalize(s, "vi")

    def test_output_is_nfc(self, norm):
        import unicodedata
        out = norm.normalize(unicodedata.normalize("NFD", "tiếng Việt có 2 thanh"), "vi")
        assert out == unicodedata.normalize("NFC", out)

    def test_nfd_routes_to_vietnamese_on_auto(self, norm):
        import unicodedata
        # no "đ" in this sentence, so detection relies on the stacked marks
        out = norm.normalize(unicodedata.normalize("NFD", "tiếng Việt có 2 thanh"), "auto")
        assert "hai" in out and "two" not in out

    @pytest.mark.parametrize("raw,expected", [
        ("xin\u200bchào", "xin chào"),  # a zero-width space BETWEEN WORDS separates
        ("xin\ufeffchào", "xin chào"),  # so does a BOM in that position
        ("xin\x00chào", "xinchào"),     # a NUL is noise, not a separator
        ("xin\xa0chào", "xin chào"),    # non-breaking space becomes a real space
        ("co\xadoperate", "cooperate"),  # a soft hyphen is an intra-word break
    ])
    def test_invisible_characters_removed(self, raw, expected):
        from confuciustts.frontend.vi_normalizer import normalize_vietnamese
        assert normalize_vietnamese(raw) == expected

    @pytest.mark.parametrize("text", [
        "", " ", "\n\t ", "...", "—", "a" * 3000, "1" * 200,
    ])
    def test_degenerate_input_does_not_crash(self, text):
        from confuciustts.frontend.vi_normalizer import normalize_vietnamese
        normalize_vietnamese(text)

    @pytest.mark.parametrize("text", [
        "Giá 1.141,5 đồng", "ngày 22/9/2026", "Rh+ và 5 ng/ml", "NV ở CH",
        "mạng 5G", "Gọi 0912 345 678", "Email a@b.com", "Phòng 25m2",
        "Brand Test v.v . Còn độ bền",   # cleanup used to create "v.v." only on pass 2
        "5%pin", "TP.Hà Nội",
    ])
    def test_idempotent(self, text):
        from confuciustts.frontend.vi_normalizer import normalize_vietnamese as f
        assert f(f(text)) == f(text)


class TestNoWelding:
    """Replacements must not weld onto the following word (ABBREVIATIONS table)."""

    @pytest.mark.parametrize("text,expected", [
        ("5%pin", "năm phần trăm pin"),
        ("20%pin", "hai mươi phần trăm pin"),
        ("TP.Hà Nội", "thành phố Hà Nội"),
        ("ở TP.Đà Nẵng", "ở thành phố Đà Nẵng"),
        ("v.v.Giải trí", "vân vân Giải trí"),
        ("Brand Test v.v . Còn", "Brand Test vân vân Còn"),
        ("tăng 50%", "tăng năm mươi phần trăm"),   # no trailing space left behind
        ("TP.HCM", "thành phố Hồ Chí Minh"),       # the compound rule still wins
    ])
    def test_abbreviations_keep_word_boundaries(self, text, expected):
        from confuciustts.frontend.vi_normalizer import normalize_vietnamese
        assert normalize_vietnamese(text) == expected


# --------------------------------------------------------------------------- vinorm parity
# One class per rule file in vinorm's RegexRule/, so the pure-Python port can be
# checked against the behaviour it replaces.


def _n(text, **kw):
    from confuciustts.frontend.vi_normalizer import normalize_vietnamese
    return normalize_vietnamese(text, **kw)


class TestVinormDates:
    @pytest.mark.parametrize("text", ["Hôm nay 22/9/2026", "ký ngày 22-9-2026", "bản 22.9.2026"])
    def test_date_1_all_separators(self, text):
        out = _n(text)
        assert "hai mươi hai tháng chín năm hai nghìn không trăm hai mươi sáu" in out

    @pytest.mark.parametrize("text,expected", [
        ("ngày 30/4 nghỉ lễ", "ngày ba mươi tháng bốn nghỉ lễ"),
        ("chiều 2/9 có lễ", "chiều hai tháng chín có lễ"),
    ])
    def test_date_3_day_month_only(self, text, expected):
        assert _n(text) == expected

    def test_date_from_to_days(self):
        assert _n("từ 20-25/3 nghỉ") == "từ hai mươi đến hai mươi lăm tháng ba nghỉ"

    def test_date_from_to_day_month(self):
        assert _n("ngày 20/3-25/3 họp").startswith("ngày hai mươi tháng ba đến hai mươi lăm tháng ba")

    def test_month_year(self):
        assert _n("tháng 9/2026") == "tháng chín năm hai nghìn không trăm hai mươi sáu"

    def test_month_range(self):
        assert _n("từ 01-05/2020").startswith("từ tháng một đến tháng năm năm")

    def test_no_double_ngay(self):
        assert _n("ngày 22/9/2026").count("ngày") == 1


class TestVinormTime:
    @pytest.mark.parametrize("text,expected", [
        ("họp 14h", "họp mười bốn giờ"),
        ("đến 9:30", "đến chín giờ ba mươi phút"),
        ("Lúc 9h30 sáng", "Lúc chín giờ ba mươi phút sáng"),
        ("9h30 pm", "chín giờ ba mươi phút chiều"),
    ])
    def test_time(self, text, expected):
        assert _n(text) == expected


class TestVinormPhone:
    @pytest.mark.parametrize("text,expected", [
        ("Gọi 0912 345 678", "Gọi không chín một hai ba bốn năm sáu bảy tám"),
        ("Hotline 1900 636 839", "Hotline một chín không không sáu ba sáu tám ba chín"),
    ])
    def test_digit_by_digit(self, text, expected):
        assert _n(text) == expected

    def test_paren_area_code_balanced(self):
        out = _n("liên hệ (024) 3825-1234")
        assert out.count("(") == out.count(")") and "không hai bốn" in out


class TestVinormFootball:
    @pytest.mark.parametrize("text,expected", [
        ("tỉ số 3-1", "tỉ số ba một"),
        ("đội hình 4-4-2", "đội hình bốn bốn hai"),
        ("đội U23 thắng", "đội u hai mươi ba thắng"),
        ("U.19 Việt Nam", "u mười chín Việt Nam"),
    ])
    def test_football(self, text, expected):
        assert _n(text) == expected


class TestVinormPoliticalDivision:
    @pytest.mark.parametrize("text,expected", [
        ("Q.1 TP.HCM", "quận một thành phố Hồ Chí Minh"),
        ("P.5 Q.Bình Thạnh", "phường năm quận Bình Thạnh"),
        ("H.Bình Chánh", "huyện Bình Chánh"),
        ("TX.Thuận An", "thị xã Thuận An"),
        ("X.An Phú", "xã An Phú"),
        ("KP.3", "khu phố ba"),
    ])
    def test_political(self, text, expected):
        assert _n(text) == expected


class TestVinormRoman:
    @pytest.mark.parametrize("text,expected", [
        ("thứ III", "thứ ba"), ("thứ IV", "thứ tư"), ("kỷ XX", "kỷ hai mươi"),
        ("khóa IV", "khóa bốn"), ("quý II", "quý hai"), ("lần thứ XV", "lần thứ mười lăm"),
    ])
    def test_roman(self, text, expected):
        assert _n(text) == expected

    def test_roman_helper(self):
        from confuciustts.frontend.vi_normalizer import roman_to_int
        assert roman_to_int("XIV") == 14 and roman_to_int("MMM") is None


class TestVinormStreetOffice:
    @pytest.mark.parametrize("text,expected", [
        ("số 15A Lê Lợi", "số một năm a Lê Lợi"),
        ("phòng 12B", "phòng một hai bê"),
        ("đường 30/4", "đường ba mươi tháng bốn"),
        ("ngõ 25/3", "ngõ hai mươi lăm tháng ba"),
    ])
    def test_codes_and_fractions(self, text, expected):
        assert _n(text) == expected


class TestVinormCurrencyAndUnits:
    @pytest.mark.parametrize("text,expected", [
        ("giá 5$", "giá năm đô la"),
        ("$5 thôi", "năm đô la thôi"),       # symbol first still reads number first
        ("100€", "một trăm ê rô"),
        ("50%", "năm mươi phần trăm"),
        ("5Ω", "năm ôm"),                      # table key is lower case "ω"
        ("2.000.000₩", "hai triệu won"),
    ])
    def test_currency(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("250 mg", "hai trăm năm mươi mi li gam"),
        ("25m2", "hai mươi lăm mét vuông"),
        ("31°C", "ba mươi mốt độ xê"),
        ("100 km/h", "một trăm ki lô mét một giờ"),
        ("5 ng/ml", "năm nanogam trên mi li lít"),
        ("350 kWh", "ba trăm năm mươi ki lô oát giờ"),
        ("220V", "hai trăm hai mươi vôn"),
        ("10 µm", "mười mi cờ rô mét"),        # MICRO SIGN, not GREEK MU
    ])
    def test_units(self, text, expected):
        assert _n(text) == expected

    def test_si_prefix_not_applied_to_acronyms(self):
        # "DHA" must not compose as d + ha -> "đề xi héc ta"
        assert "DHA" in _n("MenaQ7 K2 DHA")


class TestVinormNumbers:
    @pytest.mark.parametrize("text,expected", [
        ("1.234.567", "một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy"),
        ("1,234,567", "một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy"),
        ("1.234,5", "một nghìn hai trăm ba mươi tư phẩy năm"),
        ("1,234.5", "một nghìn hai trăm ba mươi tư phẩy năm"),
        ("3,5", "ba phẩy năm"),
        ("3.1", "ba phẩy một"),          # a lone dot is a decimal point
        ("USB 3.1", "USB ba chấm một"),  # unless a version word introduces it
        ("1.2.3", "một chấm hai chấm ba"),
    ])
    def test_four_formats(self, text, expected):
        assert _n(text) == expected


class TestVinormTeencode:
    @pytest.mark.parametrize("text,expected", [
        ("ko biết gì", "không biết gì"),
        ("bùl quá", "buồn quá"),
        ("lướt net", "lướt nét"),            # loanword reading, kept on purpose
    ])
    def test_teencode(self, text, expected):
        assert _n(text) == expected

    def test_teencode_off(self):
        assert _n("ko biết gì", teencode=False) == "ko biết gì"

    def test_teencode_is_lowercase_only(self):
        # "uc" -> "úc" in the table must not touch the ingredient "UC-II"
        assert _n("UC-II không biến tính") == "UC-II không biến tính"

    def test_english_past_tense_kept(self):
        assert "won" in _n("Team won 3-1")


class TestVinormOptIn:
    def test_spell_acronyms_off_by_default(self):
        assert _n("VTV đưa tin") == "VTV đưa tin"

    def test_spell_acronyms_on(self):
        assert _n("VTV đưa tin", spell_acronyms=True) == "vê tê vê đưa tin"

    def test_spell_letters_helper(self):
        from confuciustts.frontend.vi_normalizer import spell_letters
        assert spell_letters("VTV") == "vê tê vê"
        assert spell_letters("GPT", english=True) == "giy pi ti"

    def test_read_codes_opt_in(self):
        assert _n("iPhone 15", read_codes=False) == "iPhone mười lăm"
        assert "một năm" in _n("iPhone 15", read_codes=True)


class TestNoVinormDependency:
    def test_module_does_not_import_vinorm(self):
        import pathlib
        src = pathlib.Path("confuciustts/frontend/vi_normalizer.py").read_text(encoding="utf-8")
        assert "import vinorm" not in src and "TTSnorm" not in src

    def test_all_tables_present(self):
        from confuciustts.frontend import vi_normalizer as m
        assert len(m._ACRONYMS) > 3000 and len(m._TEENCODE) > 400
        assert len(m._POPULAR) > 7000 and len(m._LETTER_VN) > 25
        assert len(m._UNIT_MAP) > 100 and len(m._PREFIX_UNIT) > 15


class TestHeightAndDuration:
    """Vietnamese writes heights as "1m20" (1 metre 20) and durations as "7h 30 phút"."""

    @pytest.mark.parametrize("text,expected", [
        ("1m20", "một mét hai mươi"),            # not "m2" + "0"
        ("cao 1m20", "cao một mét hai mươi"),
        ("1m75", "một mét bảy mươi lăm"),
        ("1m", "một mét"),
        ("cao 1 m 20", "cao một mét hai mươi"),
    ])
    def test_height(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("25m2", "hai mươi lăm mét vuông"),      # a real exponent still reads
        ("5m3", "năm mét khối"),
        ("Phòng 25m2", "Phòng hai mươi lăm mét vuông"),   # area, not a room code
    ])
    def test_exponent_still_works(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("7h 30 phút", "bảy giờ ba mươi phút"),  # the explicit "phút" is consumed once
        ("7h30 phút", "bảy giờ ba mươi phút"),
        ("chạy hết 7h 30 phút", "chạy hết bảy giờ ba mươi phút"),
        ("7h30", "bảy giờ ba mươi phút"),
        ("7h", "bảy giờ"),
    ])
    def test_duration_no_double_phut(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("phòng 12B", "phòng một hai bê"),       # room code still read character by character
        ("lớp 10A1", "lớp một không a một"),
    ])
    def test_room_codes_unaffected(self, text, expected):
        assert _n(text) == expected


class TestIdentifierCodes:
    """vinorm Codenumber, context-gated. An id is read character by character; a
    quantity is not. "Mã BHYT: 4413219578" is an identifier, not four billion."""

    @pytest.mark.parametrize("text,expected", [
        ("Mã BHYT: 4413219578", "Mã bảo hiểm y tế: bốn bốn một ba hai một chín năm bảy tám"),
        ("Mã BHXH 1234567890", "Mã bảo hiểm xã hội một hai ba bốn năm sáu bảy tám chín không"),
        ("CMND: 012345678", "chứng minh nhân dân: không một hai ba bốn năm sáu bảy tám"),
        ("biển số 51A-12345", "biển số năm một a một hai ba bốn năm"),
    ])
    def test_codes_read_character_by_character(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text", ["IMEI 356938035643809", "STK 19001234567890",
                                      "mã số thuế 0312345678", "Mã đơn hàng 4413219578"])
    def test_no_magnitude_for_codes(self, text):
        out = _n(text)
        for w in ("tỷ", "triệu", "nghìn"):
            assert w not in out, f"{w!r} in {out!r}"

    @pytest.mark.parametrize("text,expected", [
        ("giá 1.250.000 đồng", "giá một triệu hai trăm năm mươi nghìn đồng"),
        ("giá 2000000", "giá hai triệu"),                 # round -> quantity
        ("1000000 đồng", "một triệu đồng"),
        ("dân số 97000000 người", "dân số chín mươi bảy triệu người"),
        ("năm 2026", "năm hai nghìn không trăm hai mươi sáu"),
        ("5000 mAh", "năm nghìn mi li am pe giờ"),
        ("iPhone 15", "iPhone mười lăm"),
    ])
    def test_quantities_unaffected(self, text, expected):
        assert _n(text) == expected

    def test_address_still_wins_for_addresses(self):
        assert _n("số 15A Lê Lợi") == "số một năm a Lê Lợi"
        assert _n("đường 30/4") == "đường ba mươi tháng bốn"

    def test_measurement_beats_code_context(self):
        assert _n("mã 25m2") == "mã hai mươi lăm mét vuông"



class TestIdentifierIntroducers:
    """"Ma BHYT" and "So BHYT" name the same card, so both forms must read the code
    out, in any case the user types them in."""

    @pytest.mark.parametrize("lead", ["Mã", "Số", "Số thẻ", "Mã số", "số", "mã"])
    def test_so_and_ma_are_interchangeable(self, lead):
        out = _n(f"{lead} BHYT: 4413219578")
        assert "bảo hiểm y tế" in out
        assert "bốn bốn một ba hai một chín năm bảy tám" in out
        assert "tỷ" not in out

    def test_lowercase_identifier_phrase(self):
        assert _n("số bhyt 4413219578") == "số bảo hiểm y tế bốn bốn một ba hai một chín năm bảy tám"

    @pytest.mark.parametrize("word,expected", [
        ("cmnd", "chứng minh nhân dân"),
        ("tphcm", "thành phố hồ chí minh"),
        ("ubnd", "ủy ban nhân dân"),
        ("cskh", "chăm sóc khách hàng"),
    ])
    def test_lowercase_long_acronyms_expand(self, word, expected):
        assert expected in _n(f"gửi {word} ngay")

    @pytest.mark.parametrize("text", [
        "the cand telt test",                      # the only two English-word collisions
        "Team đang train model deep learning",
        "check mail update file deadline",
        "deploy code production review build",
    ])
    def test_english_spans_are_never_expanded(self, text):
        assert _n(text) == text

    @pytest.mark.parametrize("text", ["camera chụp ảnh rất nhanh", "pin dùng được lâu hơn"])
    def test_vietnamese_words_are_never_expanded(self, text):
        assert _n(text) == text

    @pytest.mark.parametrize("word,expected", [
        ("ttbh", "trung tâm bảo hành"),       # not "thị trấn bỏ hoang"
        ("tgdt", "thế giới di động"),         # not "tôn giáo dân tộc"
        ("tggd", "thế giới di động"),         # not "trị giá giao dịch"
        ("ttdd", "trung tâm di động"),
    ])
    def test_retail_senses_beat_the_table(self, word, expected):
        assert expected in _n(f"nhân viên {word} rất nhiệt tình")
        assert expected in _n(f"nhân viên {word.upper()} rất nhiệt tình")


class TestCompoundMeasures:
    """A digit after a unit letter is an exponent only when the token ends there.
    "25m2" is an area; in "2m3cm" and "1m20" the digit is a quantity of its own."""

    @pytest.mark.parametrize("text,expected", [
        ("2m3cm", "hai mét ba xen ti mét"),
        ("cao 2m3cm", "cao hai mét ba xen ti mét"),
        ("1m2cm", "một mét hai xen ti mét"),
        ("5kg200g", "năm ki lô gam hai trăm gam"),
        ("1km500m", "một ki lô mét năm trăm mét"),
        ("1m20", "một mét hai mươi"),
        ("1m75", "một mét bảy mươi lăm"),
    ])
    def test_compound_keeps_every_digit(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("25m2", "hai mươi lăm mét vuông"),
        ("nhà 25m2", "nhà hai mươi lăm mét vuông"),
        ("5m3 nước", "năm mét khối nước"),
        ("120km2", "một trăm hai mươi ki lô mét vuông"),
        ("1000cm3", "một nghìn xen ti mét khối"),
    ])
    def test_exponents_still_read(self, text, expected):
        assert _n(text) == expected

    def test_no_digit_is_ever_dropped(self):
        for text in ["2m3cm", "1m2cm", "2m3cm rộng 1m50", "5kg200g", "1km500m"]:
            assert not re.search(r"\d", _n(text)), _n(text)


class TestMinuteMarker:
    """"1h30m" is half past one, not thirty metres past one."""

    @pytest.mark.parametrize("text,expected", [
        ("1h30m", "một giờ ba mươi phút"),
        ("2h30m", "hai giờ ba mươi phút"),
        ("7h 30 phút", "bảy giờ ba mươi phút"),
        ("9h30", "chín giờ ba mươi phút"),
        ("14h45", "mười bốn giờ bốn mươi lăm phút"),
    ])
    def test_minutes_read_as_minutes(self, text, expected):
        assert _n(text) == expected

    def test_phut_is_not_doubled(self):
        assert _n("7h 30 phút").count("phút") == 1

    def test_following_word_keeps_its_m(self):
        assert _n("9h30 mai") == "chín giờ ba mươi phút mai"
        assert _n("9h30 sáng") == "chín giờ ba mươi phút sáng"



class TestSignAndRange:
    """A sign or a relation that is never spoken is information lost: "-3.5" must not
    become "3.5", and "10-15 phút" must not become two unrelated numbers."""

    @pytest.mark.parametrize("text,expected", [
        ("-5", "âm năm"),
        ("-10°C", "âm mười độ xê"),
        ("nhiệt độ -5 độ", "nhiệt độ âm năm độ"),
        ("từ -3.5 xuống -2.0", "từ âm ba phẩy năm xuống âm hai phẩy không"),
        ("từ -10 đến 40", "từ âm mười đến bốn mươi"),
    ])
    def test_minus_reads_am(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("15-30 Gram", "mười lăm đến ba mươi Gram"),
        ("96-98%", "chín mươi sáu đến chín mươi tám phần trăm"),
        ("12-18 tháng", "mười hai đến mười tám tháng"),
        ("10-15 phút", "mười đến mười lăm phút"),
        ("1,5-2 ATP", "một phẩy năm đến hai ATP"),
        ("15%-100%", "mười lăm phần trăm đến một trăm phần trăm"),
        ("từ 10-20 mg", "từ mười đến hai mươi mi li gam"),
    ])
    def test_hyphen_between_numbers_reads_den(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("tỉ số 3-1", "tỉ số ba một"),                 # football rule claims it first
        ("đội hình 4-4-2", "đội hình bốn bốn hai"),    # a third number blocks the range
        ("từ 20-25/3", "từ hai mươi đến hai mươi lăm tháng ba"),
        ("biển số 51A-12345", "biển số năm một a một hai ba bốn năm"),
        ("0912-345-678", "không chín một hai ba bốn năm sáu bảy tám"),
    ])
    def test_other_rules_keep_their_hyphens(self, text, expected):
        assert _n(text) == expected

    def test_transcribed_subtraction_is_left_alone(self):
        out = _n("lấy cái lớn trừ cái nhỏ là 5 - 0")
        assert "đến" not in out


class TestDecimalSeparator:
    """"." is a decimal point by default and a version dot after a version word."""

    @pytest.mark.parametrize("text,expected", [
        ("hemoglobin 13.9", "hemoglobin mười ba phẩy chín"),
        ("chỉ số 102.2", "chỉ số một trăm lẻ hai phẩy hai"),
        ("6.67 inch", "sáu phẩy sáu bảy in"),
        ("giá 2.5 triệu", "giá hai phẩy năm triệu"),
    ])
    def test_plain_numbers_use_phay(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text", ["USB 3.0", "iOS 18.2", "MIUI 12.5", "Nokia 5.3",
                                      "bản 2.0", "VOS 3.0"])
    def test_version_words_use_cham(self, text):
        assert "chấm" in _n(text)
        assert "phẩy" not in _n(text)

    def test_dotted_chains_stay_chains(self):
        assert _n("1.2.3") == "một chấm hai chấm ba"
        assert _n("192.168.1.1") == ("một trăm chín mươi hai chấm một trăm sáu mươi tám "
                                     "chấm một chấm một")

    def test_thousand_groups_are_not_decimals(self):
        assert _n("1.234") == "một nghìn hai trăm ba mươi tư"
        assert _n("1.234.567") == "một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy"


class TestNumericLookalikes:
    """Unicode files superscripts, circled numerals and vulgar fractions as numbers
    but not as digits, so "\\w" treats them as letters. They must never reach the
    model unspoken, and never glue themselves to a word."""

    @pytest.mark.parametrize("text,expected", [
        ("nhà 25m²", "nhà hai mươi lăm mét vuông"),
        ("thể tích 5m³", "thể tích năm mét khối"),
        ("diện tích 120km²", "diện tích một trăm hai mươi ki lô mét vuông"),
        ("100m²/người", "một trăm mét vuông trên người"),
        ("25 m²", "hai mươi lăm mét vuông"),
    ])
    def test_superscript_units_read(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("½ ¾", "một phần hai ba phần tư"),
        ("uống ½ viên", "uống một phần hai viên"),
        ("⅓ ¼ ⅔", "một phần ba một phần tư hai phần ba"),
    ])
    def test_vulgar_fractions_read(self, text, expected):
        assert _n(text) == expected

    def test_circled_numerals_and_roman_letterforms(self):
        assert _n("① ② ③") == "một hai ba"
        assert _n("thứ Ⅲ") == "thứ ba"

    def test_trademark_symbols_are_dropped(self):
        assert _n("Samsung®") == "Samsung"
        assert _n("iPhone™ 15") == "iPhone mười lăm"

    @pytest.mark.parametrize("text", ["37gah½", "gah½×+", "nhà 25m² giá 2 tỷ", "½ viên"])
    def test_lookalikes_do_not_break_idempotence(self, text):
        once = _n(text)
        assert _n(once) == once

    def test_no_numeric_lookalike_survives(self):
        out = _n("nhà 25m², thể tích 5m³, uống ½ viên, ① ② Samsung®")
        assert not re.search(r"[²³¹½¾⅓¼⅔①②®™]", out), out


class TestOutputInvariants:
    """Properties that must hold for every input, checked here on the cases that
    broke them during the audit and on the corpus sweep separately."""

    @pytest.mark.parametrize("text", [
        "a​̀°",          # zero-width space between a letter and its mark
        "37gah½",                  # a fraction glued to a word
        "ah̀°5",              # a dropped base letter orphans its mark
        "nhà 25m²",
    ])
    def test_output_is_nfc(self, text):
        out = _n(text)
        assert unicodedata.normalize("NFC", out) == out

    @pytest.mark.parametrize("text", [
        "a​̀°", "37gah½", "ah̀°5",
        "Mã BHYT: 4413219578", "2m3cm", "1h30m", "15-30 Gram", "-10°C",
    ])
    def test_idempotent(self, text):
        once = _n(text)
        assert _n(once) == once


class TestStandaloneSymbols:
    """A currency sign with no number beside it still has to be spoken."""

    @pytest.mark.parametrize("text,expected", [
        ("£", "bảng"), ("€ và ¥", "ê rô và yên"),
        ("giá 100£", "giá một trăm bảng"), ("5€", "năm ê rô"),
    ])
    def test_currency_signs_always_read(self, text, expected):
        assert _n(text) == expected

    def test_won_is_not_re_mapped_by_teencode(self):
        assert "quan" not in _n("₩1000")


class TestDateRangeConsistency:
    """Two d/m pairs joined by "đến" or a dash must read the same way on both sides.
    Before this rule "Thời gian 11/6 đến 11/7" read the first as a ratio and the
    second as a date, because only the second had a context word in front of it."""

    @pytest.mark.parametrize("text,expected", [
        ("Thời gian 11/6 đến 11/7",
         "Thời gian ngày mười một tháng sáu đến ngày mười một tháng bảy"),
        ("áp dụng 1/1 đến 31/3",
         "áp dụng ngày một tháng một đến ngày ba mươi mốt tháng ba"),
        ("lịch 15/8-20/8",
         "lịch ngày mười lăm tháng tám đến ngày hai mươi tháng tám"),
    ])
    def test_date_range_with_a_cue(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("uống 1/2 đến 3/4 viên", "uống một phần hai đến ba phần tư viên"),
        ("1/2 đến 3/4", "một phần hai đến ba phần tư"),
        ("2/3 đến 3/4 dân số", "hai phần ba đến ba phần tư dân số"),
    ])
    def test_ambiguous_pairs_read_in_parallel(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("ngày 30/4", "ngày ba mươi tháng bốn"),
        ("đến 25/12", "đến hai mươi lăm tháng mười hai"),
        ("từ 20-25/3", "từ hai mươi đến hai mươi lăm tháng ba"),
        ("180/90 mmHg", "một trăm tám mươi trên chín mươi mi li mét thủy ngân"),
    ])
    def test_single_dates_and_ratios_unchanged(self, text, expected):
        assert _n(text) == expected



class TestAuditFixes:
    """Regression cover for the defects an independent audit reproduced."""

    @pytest.mark.parametrize("text,expected", [
        ("Cho 5 g muối", "Cho năm gam muối"),          # "g" is grams, not giờ
        ("Protein 9 g", "Protein chín gam"),
        ("12 g đường", "mười hai gam đường"),
        ("lúc 8g", "lúc tám giờ"),                      # unless a clock cue is beside it
        ("8g sáng", "tám giờ sáng"),
    ])
    def test_gram_is_not_an_hour(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("Động cơ điện 3 pha", "Động cơ điện ba pha"),  # not "pi cô héc ta"
        ("mua 2 nha", "mua hai nha"),
        ("5 cha con", "năm cha con"),
        ("400 IU vitamin D", "bốn trăm đơn vị quốc tế vitamin D"),  # upper case still a unit
        ("5 ha đất", "năm héc ta đất"),
        ("Họp lúc 5 pm", "Họp lúc năm giờ chiều"),
        ("Hẹn 7 am nhé", "Hẹn bảy giờ sáng nhé"),
    ])
    def test_real_words_are_not_si_units(self, text, expected):
        assert _n(text) == expected

    def test_no_crash_on_a_huge_digit_run(self):
        for text in ["Số " + "1" * 4997 + "000", "1." + ".".join(["000"] * 1500)]:
            assert _n(text)                              # used to raise ValueError

    def test_long_paste_is_linear(self):
        import time
        text = ("Giá 1.250.000 đồng và 15-30 Gram mỗi ngày. " * 2300)[:100000]
        t0 = time.time()
        _n(text)
        assert time.time() - t0 < 5.0                    # was ~57 s

    @pytest.mark.parametrize("text,expected", [
        ("10³ m", "mười mũ ba m"),                      # a power, not 103
        ("2³ = 8", "hai mũ ba bằng tám"),
        ("nhà 25m²", "nhà hai mươi lăm mét vuông"),     # a unit exponent
    ])
    def test_superscript_power_versus_exponent(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("Lợi nhuận -12%", "Lợi nhuận âm mười hai phần trăm"),
        ("tăng trưởng -3,5%", "tăng trưởng âm ba phẩy năm phần trăm"),
        ("-5$", "âm năm đô la"),
    ])
    def test_sign_survives_percent_and_currency(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text", ["bằng lái hạng C", "mục C của hợp đồng", "hạng X",
                                      "lần IIII"])
    def test_letter_grades_are_not_roman_numerals(self, text):
        assert _n(text) == text

    @pytest.mark.parametrize("text,expected", [("thứ III", "thứ ba"), ("thế kỷ XX", "thế kỷ hai mươi"),
                                               ("khóa IV", "khóa bốn"), ("thứ III4", "thứ ba bốn")])
    def test_real_roman_numerals_still_read(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text", ["Nguyễn Văn H.Đã bị bắt", "bị can P.Đã bị khởi tố",
                                      "ông Q.Và vợ", "anh X.Nói rằng"])
    def test_press_initials_are_not_divisions(self, text):
        assert _n(text) == text

    @pytest.mark.parametrize("text,expected", [
        ("Q.1 TP.HCM", "quận một thành phố Hồ Chí Minh"),
        ("H.Bình Chánh", "huyện Bình Chánh"),
        ("X.Đông Hòa", "xã Đông Hòa"),
        ("KP.4", "khu phố bốn"),
    ])
    def test_real_divisions_still_read(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("Pin 5000mAh", "Pin năm nghìn mi li am pe giờ"),
        ("dung lượng pin 4500 mAh", "dung lượng pin bốn nghìn năm trăm mi li am pe giờ"),
        ("pin 1500", "pin một nghìn năm trăm"),
    ])
    def test_battery_is_not_a_pin_code(self, text, expected):
        assert _n(text) == expected

    def test_uppercase_pin_is_still_a_code(self):
        assert _n("PIN 123456") == "PIN một hai ba bốn năm sáu"

    @pytest.mark.parametrize("text,expected", [
        ("½ kg", "một phần hai ki lô gam"),             # the unit must still be spoken
        ("2½ kg", "hai và một phần hai ki lô gam"),
        ("3/4", "ba phần tư"),
        ("tỉ lệ 1/3", "tỉ lệ một phần ba"),
        ("180/90 mmHg", "một trăm tám mươi trên chín mươi mi li mét thủy ngân"),
        ("7/9 người", "bảy trên chín người"),
    ])
    def test_fractions_versus_ratios(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [("và/hoặc", "và hoặc"), ("nam/nữ", "nam nữ")])
    def test_slash_between_words_is_dropped(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text", ["a lot of people", "my lip balm", "the fog is thick",
                                      "oh my god", "That slay"])
    def test_chat_shorthand_leaves_english_alone(self, text):
        assert _n(text) == text

    @pytest.mark.parametrize("text,expected", [("ko biết", "không biết"), ("mua net", "mua nét"),
                                               ("hàng hot", "hàng hót")])
    def test_vietnamese_shorthand_still_expands(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("007", "không không bảy"),
        ("phòng 007", "phòng không không bảy"),
        ("áo số 07", "áo số không bảy"),
    ])
    def test_leading_zeros_are_kept(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("dân số 98168833 người",
         "dân số chín mươi tám triệu một trăm sáu mươi tám nghìn tám trăm ba mươi ba người"),
        ("có 1234567 người",
         "có một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy người"),
    ])
    def test_counted_quantities_keep_their_magnitude(self, text, expected):
        assert _n(text) == expected

    def test_identifiers_still_spell_out(self):
        assert _n("IMEI 356938035643809").startswith("IMEI ba năm sáu")

    @pytest.mark.parametrize("text,expected", [
        ("12/25/2020", "ngày hai mươi lăm tháng mười hai năm hai nghìn không trăm hai mươi"),
        ("31/12/2026", "ngày ba mươi mốt tháng mười hai năm hai nghìn không trăm hai mươi sáu"),
    ])
    def test_date_fields_are_validated(self, text, expected):
        assert _n(text) == expected

    @pytest.mark.parametrize("text", ["1/0/2020", "39/12/2020", "tháng 19/2026"])
    def test_impossible_dates_are_not_dates(self, text):
        assert "tháng không" not in _n(text)
        assert "ba mươi chín tháng" not in _n(text)

    @pytest.mark.parametrize("text,expected", [
        ("5-3=2", "năm-ba bằng hai"), ("góc 45°", "góc bốn mươi lăm độ"),
        ("giá 100k", "giá một trăm nghìn"), ("giá 2tr5", "giá hai triệu năm"),
        ("100 ‰", "một trăm phần nghìn"), ("nhiệt độ 37 ℃", "nhiệt độ ba mươi bảy độ xê"),
    ])
    def test_assorted_unspoken_tokens(self, text, expected):
        assert _n(text) == expected

    def test_email_does_not_split_the_word_before_it(self):
        assert "hộpthưtuan" in _n("hộpthưtuan@gmail.com")
        assert _n("Liên hệ abc@gmail.com nhé") == "Liên hệ abc@gmail.com nhé"



class TestZeroWidthSeparators:
    """An invisible character between two alphanumerics separates them. Deleting it
    welded two words into one token, and between digits it invented a new number."""

    @pytest.mark.parametrize("raw", ["tiếng\u200bViệt", "tiếng\u200cViệt", "tiếng\u200dViệt",
                                     "tiếng\u2060Việt", "tiếng\ufeffViệt"])
    def test_words_are_not_welded(self, raw):
        assert _n(raw) == "tiếng Việt"

    def test_several_in_one_line(self):
        assert _n("Xin\u200bchào\u200bbạn") == "Xin chào bạn"

    def test_no_magnitude_is_invented_between_digits(self):
        assert _n("12\u200b3") == "mười hai ba"          # not "một trăm hai mươi ba"

    def test_a_unit_still_attaches(self):
        assert _n("100\u200bkg") == "một trăm ki lô gam"

    def test_line_separators_are_whitespace(self):
        assert _n("dòng1\u2028dòng2") == "dòng một dòng hai"

    @pytest.mark.parametrize("raw,expected", [
        ("a\u200b\u0300", "à"),                          # must still be deleted, so NFC
        ("a\u200b\u0300\u00b0", "à\u00b0"),              # composes rather than separates
    ])
    def test_deleted_before_a_combining_mark(self, raw, expected):
        out = _n(raw)
        assert out == expected
        assert unicodedata.normalize("NFC", out) == out

    def test_no_invisible_character_survives(self):
        out = _n("tiếng\u200bViệt\u2060có\u00ad2\u180ethanh")
        assert not re.search(r"[\u00ad\u180e\u200b-\u200f\u2028\u2029\u2060-\u2064\ufeff]", out)
