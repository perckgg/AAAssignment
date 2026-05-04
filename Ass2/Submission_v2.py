# --- Cell 0 ---
# Change your team name to Group_xx
TEAM_NAME = 'Group_01'

# --- Cell 1 ---
# NOTE: you CAN change this cell
# If you want to use your own database, download it here
# !gdown ...

# --- Cell 2 ---
# NOTE: you CAN change this cell
# you must place ALL pip install here

# --- Cell 3 ---
# NOTE: you CAN change this cell
# import your library here
import time
import re
from unidecode import unidecode
from Levenshtein import distance as lev_distance

# --- Cell 4 ---
# NOTE: you MUST change this cell
# New methods / functions must be written under class Solution.
import re
from unidecode import unidecode
from Levenshtein import distance as lev_distance

class Solution:
    def __init__(
        self,
        list_province: list[str],
        list_ward: list[str],
        list_street: list[str]
    ):
        self.list_province = list_province
        self.list_ward = list_ward
        self.list_street = list_street

        self.province_lower = {p.lower(): p for p in list_province}
        self.ward_lower = {w.lower(): w for w in list_ward}
        self.street_lower = {s.lower(): s for s in list_street}

        # Pre-compute ASCII versions for all items
        self.province_ascii = {unidecode(p).lower(): p for p in list_province}
        self.ward_ascii = {unidecode(w).lower(): w for w in list_ward}
        self.street_ascii = {unidecode(s).lower(): s for s in list_street}

        # Pre-compute ASCII list for edit distance (avoid recomputing)
        self.province_ascii_list = [(unidecode(p).lower(), p) for p in list_province]
        self.ward_ascii_list = [(unidecode(w).lower(), w) for w in list_ward]
        self.street_ascii_list = [(unidecode(s).lower(), s) for s in list_street]

        self.province_prefixes = [
            'thành phố ', 'thanh pho ', 'tỉnh ', 'tinh ',
            'tp.', 'tp ', 't.', 't '
        ]
        self.ward_prefixes = [
            'phường', 'phuong ', 'p.'
        ]
        self.street_prefixes = [
            'đường ', 'duong ', 'phố ', 'pho '
        ]

    def _normalize_text(self, s):
        return s.strip()

    def _remove_province_prefix(self, text):
        text = text.strip()
        text_lower = text.lower()
        for prefix in self.province_prefixes:
            if text_lower.startswith(prefix):
                return text[len(prefix):].strip()
        return text

    def _remove_ward_prefix(self, text):
        text = text.strip()
        text_lower = text.lower()
        for prefix in self.ward_prefixes:
            if text_lower.startswith(prefix):
                result = text[len(prefix):].strip()
                if result and result[0] == '.':
                    result = result[1:].strip()
                return result
        if len(text) > 1 and text[0] in 'Pp' and text[1].isupper():
            return text[1:]
        return text

    def _remove_street_prefix(self, text):
        text = text.strip()
        text_lower = text.lower()
        for prefix in self.street_prefixes:
            if text_lower.startswith(prefix):
                return text[len(prefix):].strip()
        return text

    def _best_by_edit_distance(self, query_ascii, candidates_ascii_list, max_dist=3):
        """Find best match using edit distance. Only considers candidates
        within max_dist edits and with length difference <= max_dist."""
        best = None
        best_dist = max_dist + 1
        query_len = len(query_ascii)

        for cand_ascii, cand_original in candidates_ascii_list:
            # Quick length filter - skip if length difference too large
            if abs(len(cand_ascii) - query_len) > max_dist:
                continue
            d = lev_distance(query_ascii, cand_ascii)
            if d < best_dist:
                best_dist = d
                best = cand_original
                if d == 0:
                    return best  # exact match

        return best

    def _match_province(self, text):
        text = self._normalize_text(text)
        cleaned = self._remove_province_prefix(text)

        cleaned_lower = cleaned.lower()
        if cleaned_lower.startswith('tp') and len(cleaned) > 2 and cleaned[2] != ' ' and cleaned[2] != '.':
            cleaned = cleaned[2:]
        elif cleaned_lower.startswith('t') and len(cleaned) > 1 and cleaned[1] != ' ' and cleaned[1] != '.':
            candidate = cleaned[1:]
            if candidate.lower() in self.province_lower or unidecode(candidate).lower() in self.province_ascii:
                cleaned = candidate

        if cleaned.lower() in self.province_lower:
            return self.province_lower[cleaned.lower()]

        ascii_cleaned = unidecode(cleaned).lower()
        if ascii_cleaned in self.province_ascii:
            return self.province_ascii[ascii_cleaned]

        # Substring match
        best = None
        best_len = 0
        for p in self.list_province:
            p_lower = p.lower()
            c_lower = cleaned.lower()
            if p_lower in c_lower or c_lower in p_lower:
                if len(p) > best_len:
                    best = p
                    best_len = len(p)

        if best:
            return best

        for p in self.list_province:
            p_ascii = unidecode(p).lower()
            if p_ascii in ascii_cleaned or ascii_cleaned in p_ascii:
                if len(p) > best_len:
                    best = p
                    best_len = len(p)

        if best:
            return best

        # Fallback: edit distance
        result = self._best_by_edit_distance(ascii_cleaned, self.province_ascii_list, max_dist=3)
        if result:
            return result

        return self.list_province[0]

    def _match_ward(self, text):
        text = self._normalize_text(text)
        cleaned = self._remove_ward_prefix(text)

        if cleaned.lower() in self.ward_lower:
            return self.ward_lower[cleaned.lower()]

        ascii_cleaned = unidecode(cleaned).lower()
        if ascii_cleaned in self.ward_ascii:
            return self.ward_ascii[ascii_cleaned]

        for p in self.list_province:
            if cleaned.lower().endswith(p.lower()):
                candidate = cleaned[:len(cleaned)-len(p)].strip().rstrip(',').strip()
                if candidate.lower() in self.ward_lower:
                    return self.ward_lower[candidate.lower()]

        best = None
        best_len = 0
        for w in self.list_ward:
            w_lower = w.lower()
            c_lower = cleaned.lower()
            if w_lower == c_lower:
                return w
            if w_lower in c_lower:
                if len(w) > best_len:
                    best = w
                    best_len = len(w)

        if best:
            return best

        for w in self.list_ward:
            w_ascii = unidecode(w).lower()
            if w_ascii in ascii_cleaned or ascii_cleaned in w_ascii:
                if len(w) > best_len:
                    best = w
                    best_len = len(w)

        if best:
            return best

        # Fallback: edit distance
        result = self._best_by_edit_distance(ascii_cleaned, self.ward_ascii_list, max_dist=3)
        if result:
            return result

        return self.list_ward[0]

    def _match_street(self, text):
        text = self._normalize_text(text)

        cleaned = re.sub(r'^\d+(/\d+[A-Za-z]*)*\s+', '', text).strip()

        cleaned = re.sub(r'^[Nn]gõ\s+[\d/]+[A-Za-z]*\s*', '', cleaned).strip()
        cleaned = re.sub(r'^[Hh]ẻm\s+[\d/]+[A-Za-z]*\s*', '', cleaned).strip()

        cleaned = self._remove_street_prefix(cleaned)

        cleaned = re.sub(r'^\d+(/\d+[A-Za-z]*)*\s+', '', cleaned).strip()

        if re.match(r'^[Ss]ố\s+\d+', cleaned):
            cleaned = 'Số' + cleaned[2:]

        cleaned = re.sub(r'^[Qq]uốc\s+[Ll]ộ', 'Quốc Lộ', cleaned)

        if cleaned.lower() in self.street_lower:
            return self.street_lower[cleaned.lower()]

        ascii_cleaned = unidecode(cleaned).lower()
        if ascii_cleaned in self.street_ascii:
            return self.street_ascii[ascii_cleaned]

        cleaned_normalized = ' '.join(cleaned.lower().split())
        for s in self.list_street:
            if ' '.join(s.lower().split()) == cleaned_normalized:
                return s

        best = None
        best_len = 0
        c_lower = cleaned.lower()
        for s in self.list_street:
            s_lower = s.lower()
            if s_lower == c_lower:
                return s
            if s_lower in c_lower:
                if len(s) > best_len:
                    best = s
                    best_len = len(s)

        if best and best_len >= 3:
            return best

        # ASCII substring matching - prefer closest length match
        best_diff = float('inf')
        for s in self.list_street:
            s_ascii = unidecode(s).lower()
            if s_ascii == ascii_cleaned:
                return s
            if ascii_cleaned in s_ascii:
                diff = len(s_ascii) - len(ascii_cleaned)
                if diff < best_diff:
                    best_diff = diff
                    best = s
            elif s_ascii in ascii_cleaned:
                diff = len(ascii_cleaned) - len(s_ascii)
                if diff < best_diff:
                    best_diff = diff
                    best = s

        if best:
            return best

        # Fallback: edit distance (handles missing chars anywhere)
        result = self._best_by_edit_distance(ascii_cleaned, self.street_ascii_list, max_dist=3)
        if result:
            return result

        return self.list_street[0]

    def process(self, s: str):
        s = s.strip()
        parts = [p.strip() for p in s.split(',')]
        parts = [p for p in parts if p]

        if len(parts) >= 3:
            province_text = parts[-1]
            ward_text = parts[-2]
            street_text = parts[0]
        elif len(parts) == 2:
            province_text = parts[-1]
            ward_text = parts[0]
            street_text = parts[0]
        else:
            province_text = s
            ward_text = s
            street_text = s

        province = self._match_province(province_text)
        ward = self._match_ward(ward_text)
        street = self._match_street(street_text)

        return {
            "province": province,
            "ward": ward,
            "street": street,
        }



# --- Cell 5 ---
# Download public test

# --- Cell 6 ---
def _judge_normalize(s: str, context=None):
    return s

EXCEL_FILE = f'{TEAM_NAME}.xlsx'

import json
import time
with open('test.json') as f:
    data = json.load(f)

with open("list_province.txt") as f:
    list_province = f.read().strip().splitlines()

with open("list_ward.txt") as f:
    list_ward = f.read().strip().splitlines()

with open("list_street.txt") as f:
    list_street = f.read().strip().splitlines()

summary_only = True
df = []
solution = Solution(
    list_province=list_province,
    list_ward=list_ward,
    list_street=list_street,
)
timer = []
correct = 0
for test_idx, data_point in enumerate(data):
    address = data_point["address"]

    ok = 0
    try:
        answer = data_point["result"]
        answer["province_normalized"] = _judge_normalize(answer["province"])
        answer["ward_normalized"] = _judge_normalize(answer["ward"])
        answer["street_normalized"] = _judge_normalize(answer["street"])

        start = time.perf_counter_ns()
        result = solution.process(address)
        finish = time.perf_counter_ns()
        timer.append(finish - start)
        result["province_normalized"] = _judge_normalize(result["province"])
        result["ward_normalized"] = _judge_normalize(result["ward"])
        result["street_normalized"] = _judge_normalize(result["street"])

        province_correct = int(answer["province_normalized"] == result["province_normalized"])
        ward_correct = int(answer["ward_normalized"] == result["ward_normalized"])
        street_correct = int(answer["street_normalized"] == result["street_normalized"])
        ok = province_correct + street_correct + ward_correct

        df.append([
            test_idx,
            address,

            answer["province"],
            result["province"],
            answer["province_normalized"],
            result["province_normalized"],
            province_correct,

            answer["ward"],
            result["ward"],
            answer["ward_normalized"],
            result["ward_normalized"],
            ward_correct,

            answer["street"],
            result["street"],
            answer["street_normalized"],
            result["street_normalized"],
            street_correct,

            ok,
            timer[-1] / 1_000_000_000,
        ])
    except Exception as e:
        print(f"{answer = }")
        print(f"{result = }")
        df.append([
            test_idx,
            address,

            answer["province"],
            "EXCEPTION",
            answer["province_normalized"],
            "EXCEPTION",
            0,

            answer["ward"],
            "EXCEPTION",
            answer["ward_normalized"],
            "EXCEPTION",
            0,

            answer["street"],
            "EXCEPTION",
            answer["street_normalized"],
            "EXCEPTION",
            0,

            0,
            0,
        ])
        # any failure count as a zero correct
        pass
    correct += ok


    if not summary_only:
        # responsive stuff
        print(f"Test {test_idx:5d}/{len(data):5d}")
        print(f"Correct: {ok}/3")
        print(f"Time Executed: {timer[-1] / 1_000_000_000:.4f}")


print(f"-"*30)
total = len(data) * 3
score_scale_10 = round(correct / total * 10, 2)
if len(timer) == 0:
    timer = [0]
max_time_sec = round(max(timer) / 1_000_000_000, 4)
avg_time_sec = round((sum(timer) / len(timer)) / 1_000_000_000, 4)

import pandas as pd

df2 = pd.DataFrame(
    [[correct, total, score_scale_10, max_time_sec, avg_time_sec]],
    columns=['correct', 'total', 'score / 10', 'max_time_sec', 'avg_time_sec',],
)

columns = [
    'ID',
    'text',

    'province',
    'province_student',
    'province_normalized',
    'province_student_normalized',
    'province_correct',

    'ward',
    'ward_student',
    'ward_normalized',
    'ward_student_normalized',
    'ward_correct',

    'street',
    'street_student',
    'street_normalized',
    'street_student_normalized',
    'street_correct',

    'total_correct',
    'time_sec',
]

df = pd.DataFrame(df)
df.columns = columns

print(f'{TEAM_NAME = }')
print(f'{EXCEL_FILE = }')
print(df2)

writer = pd.ExcelWriter(EXCEL_FILE, engine='xlsxwriter')
df2.to_excel(writer, index=False, sheet_name='summary')
df.to_excel(writer, index=False, sheet_name='details')
writer.close()
