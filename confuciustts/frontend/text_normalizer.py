# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
# Copyright 2026 NetEase Youdao. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from https://github.com/FunAudioLLM/CosyVoice

import re
import unicodedata
import regex
import inflect
from typing import List, Callable
from functools import partial

from confuciustts.utils.text_utils import to_katakana
from confuciustts.frontend.vi_normalizer import normalize_vietnamese

# Vietnamese-specific segmentation: comma splits a Vietnamese clause, and the
# Unicode list covers every punctuation that can end a Vietnamese sentence.
_VI_SENTENCE_PUNCTUATION = ['.', '?', '!', ';', ':', '…']
_VI_CLAUSE_PUNCTUATION = [',']


class TextNormalizer:
    def __init__(self):
        self.inflect_parser = inflect.engine()

        self.chinese_pattern = re.compile(r'[一-鿿]+')
        self.punctuation_pattern = r'^[\p{P}\p{S}]*$'
        # Lowercase-only Vietnamese diacritic class; see contains_vietnamese
        # for why uppercase is handled by casefolding the input instead.
        self._vi_mark_class = re.compile(
            r'[ăâđêôơư'
            r'àáảãạằắẳẵặèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ'
            r'ầấẩẫậềếểễệồốổỗộờớởỡợừứửữự]'
        )

        self.zh_normalizer = None
        self.en_normalizer = None
        try:
            from wetext import Normalizer
            self.zh_normalizer = Normalizer(remove_erhua=False)
            self.en_normalizer = Normalizer()
        except ImportError:
            pass

    def contains_chinese(self, text: str) -> bool:
        return bool(self.chinese_pattern.search(text))

    def contains_vietnamese(self, text: str) -> bool:
        # Vietnamese tone marks that never occur in the other supported
        # languages' scripts. Covers every layer of diacritics: bare â đ ơ ư,
        # tone-marked plain vowels (à á ả ã ạ, è é ẻ ẽ ẹ, ...), and tone marks
        # stacked on already-marked vowels (ầ ấ ẩ ẫ ậ, ề ế ể ễ ệ, ồ ố ổ ỗ ộ,
        # ờ ớ ở ỡ ợ, ừ ứ ử ữ ự) — "tiếng Việt" needs the stacked ế and ệ.
        # Matching runs on a casefolded copy of the text so uppercase variants
        # (TIẾNG, HỒ) need no second class; Vietnamese casefold maps each
        # uppercase mark to exactly the lowercase form listed here.
        # Compose first: decomposed input ("e" + combining acute) carries none of
        # the precomposed characters this class matches, so NFD Vietnamese without
        # an "đ" would otherwise be routed to the English normalizer.
        return bool(self._vi_mark_class.search(unicodedata.normalize("NFC", text).casefold()))

    def is_only_punctuation(self, text: str) -> bool:
        return bool(regex.fullmatch(self.punctuation_pattern, text))

    def replace_corner_marks(self, text: str) -> str:
        text = text.replace('²', '平方')
        text = text.replace('³', '立方')
        return text

    def remove_brackets(self, text: str) -> str:
        text = text.replace('（', '').replace('）', '')
        text = text.replace('【', '').replace('】', '')
        text = text.replace('`', '').replace('`', '')
        text = text.replace("——", " ")
        return text

    def remove_blank_between_chinese(self, text: str) -> str:
        out_str = []
        for i, c in enumerate(text):
            if c == " ":
                if i > 0 and i < len(text) - 1:
                    if (text[i + 1].isascii() and text[i + 1] != " " and
                        text[i - 1].isascii() and text[i - 1] != " "):
                        out_str.append(c)
            else:
                out_str.append(c)
        return "".join(out_str)

    def spell_out_numbers(self, text: str) -> str:
        new_text = []
        start = None

        for i, c in enumerate(text):
            if not c.isdigit():
                if start is not None:
                    num_str = self.inflect_parser.number_to_words(text[start:i])
                    new_text.append(num_str)
                    start = None
                new_text.append(c)
            else:
                if start is None:
                    start = i

        if start is not None and start < len(text):
            num_str = self.inflect_parser.number_to_words(text[start:])
            new_text.append(num_str)

        return ''.join(new_text)

    def normalize_chinese(self, text: str) -> str:
        if self.zh_normalizer is not None:
            text = self.zh_normalizer.normalize(text)

        text = self.remove_blank_between_chinese(text)

        text = self.replace_corner_marks(text)

        text = text.replace(".", "。")
        text = text.replace(" - ", "，")

        text = self.remove_brackets(text)

        text = re.sub(r'[，,、]+$', '。', text)

        return text

    def normalize_english(self, text: str) -> str:
        if self.en_normalizer is not None:
            text = self.en_normalizer.normalize(text)

        text = self.spell_out_numbers(text)

        return text

    def normalize_japanese(self, text: str) -> str:
        return to_katakana(text)

    def normalize_vietnamese(self, text: str) -> str:
        return normalize_vietnamese(text)

    def normalize(self, text: str, language: str = "auto") -> str:
        if not text:
            return text

        text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        if language == "auto":
            if self.contains_chinese(text):
                language = "zh"
            elif self.contains_vietnamese(text):
                language = "vi"
            else:
                language = "en"

        if language == "zh":
            text = self.normalize_chinese(text)
        elif language == "ja":
            text = self.normalize_japanese(text)
        elif language == "vi":
            text = self.normalize_vietnamese(text)
        else:
            text = self.normalize_english(text)

        return text

    def segment_text(
        self,
        text: str,
        tokenize_fn: Callable,
        language: str = "zh",
        max_tokens: int = 80,
        min_tokens: int = 60,
        merge_threshold: int = 20,
        split_on_comma: bool = False,
    ) -> List[str]:
        # Vietnamese clauses end at commas too, and the hard-cut fallback must
        # never split a syllable in half. Everything else follows the shared path.
        is_vi = language == "vi"
        if is_vi:
            split_on_comma = True

        def calc_length(t: str) -> int:
            if language == "zh":
                return len(t)
            else:
                return len(tokenize_fn(t))

        def should_merge(t: str) -> bool:
            if language == "zh":
                return len(t) < merge_threshold
            else:
                return len(tokenize_fn(t)) < merge_threshold

        # Define punctuation marks
        if language == "zh":
            punctuation = ['。', '？', '！', '；', '：', '.', '?', '!', ';']
        elif is_vi:
            punctuation = list(_VI_SENTENCE_PUNCTUATION)
        else:
            punctuation = ['.', '?', '!', ';', ':']

        if split_on_comma:
            punctuation.extend(['，', ','])

        if text and text[-1] not in punctuation:
            text += "。" if language == "zh" else "."

        segments = []
        start = 0

        for i, char in enumerate(text):
            if char in punctuation:
                if len(text[start:i]) > 0:
                    segment = text[start:i] + char

                    if i + 1 < len(text) and text[i + 1] in ['"', '"']:
                        segment += text[i + 1]
                        start = i + 2
                    else:
                        start = i + 1

                    segments.append(segment)

        if len(segments) == 1 and calc_length(segments[0]) > max_tokens:
            long_text = segments[0][:-1]  # Remove added punctuation
            segments = []
            if is_vi:
                # Vietnamese syllables are space-separated: cut at word
                # boundaries in the same unit calc_length measures, keeping a
                # trailing space so the final join cannot glue syllables
                # together ("Mộtcâu"). Each chunk stays <= max_tokens.
                words = long_text.split(" ")
                current = ""
                for word in words:
                    candidate = current + " " + word if current else word
                    if current and calc_length(candidate) > max_tokens:
                        segments.append(current + " ")
                        current = word
                    else:
                        current = candidate
                if current:
                    segments.append(current + " ")
            else:
                for i in range(0, len(long_text), max_tokens):
                    chunk = long_text[i:i + max_tokens]
                    segments.append(chunk)

        final_segments = []
        current = ""

        for seg in segments:
            if calc_length(current + seg) > max_tokens and calc_length(current) > min_tokens:
                final_segments.append(current)
                current = ""
            current = current + seg

        if current:
            if should_merge(current) and final_segments:
                final_segments[-1] = final_segments[-1] + current
            else:
                final_segments.append(current)

        if language == "zh":
            final_segments = [
                seg[:-1] if seg and seg[-1] in ['。', '；', '：', '.', ';'] else seg
                for seg in final_segments
            ]

        final_segments = [s for s in final_segments if not self.is_only_punctuation(s)]

        return final_segments
