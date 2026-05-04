# --- Cell 0 ---
# Change your team name to Group_xx
TEAM_NAME = 'Group_12'

# --- Cell 1 ---
# Cell for downloading database/indexes if needed
# No external files required for this solution

# --- Cell 2 ---
# pip install libraries

# --- Cell 3 ---
# import libraries
import time
import re
from unidecode import unidecode
from Levenshtein import distance as lev_distance

# --- Cell 4 ---
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

        self.province_ascii = {unidecode(p).lower(): p for p in list_province}
        self.ward_ascii = {unidecode(w).lower(): w for w in list_ward}
        self.street_ascii = {unidecode(s).lower(): s for s in list_street}

        self.province_ascii_list = [(unidecode(p).lower(), p) for p in list_province]
        self.ward_ascii_list = [(unidecode(w).lower(), w) for w in list_ward]
        self.street_ascii_list = [(unidecode(s).lower(), s) for s in list_street]

        self.province_prefixes = [
            'thành phố ', 'thanh pho ',
            'tỉnh ', 'tinh ',
            'tp.', 'tp ', 't.', 't '
        ]
        self.ward_prefixes = ['phường', 'phuong ', 'p.']
        self.street_prefixes = ['đường ', 'duong ', 'phố ', 'pho ']

    def _strip_province_prefix(self, text):
        text = text.strip()
        tl = text.lower()
        for px in self.province_prefixes:
            if tl.startswith(px):
                return text[len(px):].strip()
        return text

    def _strip_ward_prefix(self, text):
        text = text.strip()
        tl = text.lower()
        for px in self.ward_prefixes:
            if tl.startswith(px):
                r = text[len(px):].strip()
                if r and r[0] == '.':
                    r = r[1:].strip()
                return r
        if len(text) > 1 and text[0] in 'Pp' and text[1].isupper():
            return text[1:]
        return text

    def _strip_street_prefix(self, text):
        text = text.strip()
        tl = text.lower()
        for px in self.street_prefixes:
            if tl.startswith(px):
                return text[len(px):].strip()
        return text

    def _find_nearest(self, query_ascii, candidates, max_dist=3):
        best = None
        best_dist = max_dist + 1
        qlen = len(query_ascii)
        for cand_ascii, cand_orig in candidates:
            if abs(len(cand_ascii) - qlen) > max_dist:
                continue
            d = lev_distance(query_ascii, cand_ascii)
            if d < best_dist:
                best_dist = d
                best = cand_orig
                if d == 0:
                    return best
        return best

    def _match_province(self, text):
        cleaned = self._strip_province_prefix(text.strip())

        cl = cleaned.lower()
        if cl.startswith('tp') and len(cleaned) > 2 and cleaned[2] not in ' .':
            cleaned = cleaned[2:]
        elif cl.startswith('t') and len(cleaned) > 1 and cleaned[1] not in ' .':
            cand = cleaned[1:]
            if cand.lower() in self.province_lower or unidecode(cand).lower() in self.province_ascii:
                cleaned = cand

        if cleaned.lower() in self.province_lower:
            return self.province_lower[cleaned.lower()]

        ac = unidecode(cleaned).lower()
        if ac in self.province_ascii:
            return self.province_ascii[ac]

        best, best_len = None, 0
        for p in self.list_province:
            pl = p.lower()
            cl2 = cleaned.lower()
            if pl in cl2 or cl2 in pl:
                if len(p) > best_len:
                    best, best_len = p, len(p)
        if best:
            return best

        for p in self.list_province:
            pa = unidecode(p).lower()
            if pa in ac or ac in pa:
                if len(p) > best_len:
                    best, best_len = p, len(p)
        if best:
            return best

        result = self._find_nearest(ac, self.province_ascii_list)
        return result if result else self.list_province[0]

    def _match_ward(self, text):
        cleaned = self._strip_ward_prefix(text.strip())

        if cleaned.lower() in self.ward_lower:
            return self.ward_lower[cleaned.lower()]

        ac = unidecode(cleaned).lower()
        if ac in self.ward_ascii:
            return self.ward_ascii[ac]

        best, best_len = None, 0
        cl = cleaned.lower()
        for w in self.list_ward:
            wl = w.lower()
            if wl == cl:
                return w
            if wl in cl:
                if len(w) > best_len:
                    best, best_len = w, len(w)
        if best:
            return best

        for w in self.list_ward:
            wa = unidecode(w).lower()
            if wa in ac or ac in wa:
                if len(w) > best_len:
                    best, best_len = w, len(w)
        if best:
            return best

        result = self._find_nearest(ac, self.ward_ascii_list)
        return result if result else self.list_ward[0]

    def _match_street(self, text):
        text = text.strip()
        cleaned = re.sub(r'^\d+(/\d+[A-Za-z]*)*\s+', '', text).strip()
        cleaned = re.sub(r'^[Nn]gõ\s+[\d/]+[A-Za-z]*\s*', '', cleaned).strip()
        cleaned = re.sub(r'^[Hh]ẻm\s+[\d/]+[A-Za-z]*\s*', '', cleaned).strip()
        cleaned = self._strip_street_prefix(cleaned)
        cleaned = re.sub(r'^\d+(/\d+[A-Za-z]*)*\s+', '', cleaned).strip()

        if re.match(r'^[Ss]ố\s+\d+', cleaned):
            cleaned = 'Số' + cleaned[2:]
        cleaned = re.sub(r'^[Qq]uốc\s+[Ll]ộ', 'Quốc Lộ', cleaned)

        if cleaned.lower() in self.street_lower:
            return self.street_lower[cleaned.lower()]

        ac = unidecode(cleaned).lower()
        if ac in self.street_ascii:
            return self.street_ascii[ac]

        best, best_len = None, 0
        cl = cleaned.lower()
        for s in self.list_street:
            sl = s.lower()
            if sl == cl:
                return s
            if sl in cl and len(s) > best_len:
                best, best_len = s, len(s)
        if best and best_len >= 3:
            return best

        best_diff = float('inf')
        best = None
        for s in self.list_street:
            sa = unidecode(s).lower()
            if sa == ac:
                return s
            if ac in sa:
                d = len(sa) - len(ac)
                if d < best_diff:
                    best_diff, best = d, s
            elif sa in ac:
                d = len(ac) - len(sa)
                if d < best_diff:
                    best_diff, best = d, s
        if best:
            return best

        result = self._find_nearest(ac, self.street_ascii_list)
        return result if result else self.list_street[0]

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

        return {
            "province": self._match_province(province_text),
            "ward": self._match_ward(ward_text),
            "street": self._match_street(street_text),
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
