"""
Hard test cases for the address classification solution.
These simulate the types of corruption/variation seen in real data
and cover edge cases that might trip up the solution.

Usage: py test_hard_cases.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json

# Hard test cases: each has 'address' and expected 'result'
# Categories of difficulty:
# 1. Missing/corrupted first character
# 2. No spaces after prefix (e.g., "PhườngX", "PX", "TPX")
# 3. Abbreviated prefixes (T, TP, P, P.)
# 4. Deep nested house numbers (e.g., "111/138/98/45")
# 5. "Đường số X" variants
# 6. No prefix at all (just street name, ward, province)
# 7. Extra whitespace / missing commas
# 8. Ward/province name overlap with street name
# 9. Very short street names ("1", "12", "B5")
# 10. "Quốc lộ" / "Quốc Lộ" variants

hard_cases = [
    # --- Category 1: Missing/corrupted first character ---
    {
        "address": "259 oàn Hoàng Minh, Sơn Đông, Vĩnh Long",
        "result": {"province": "Vĩnh Long", "ward": "Sơn Đông", "street": "Đoàn Hoàng Minh"}
    },
    {
        "address": "ặng Dung, Đồng Sơn, Quảng Trị",
        "result": {"province": "Quảng Trị", "ward": "Đồng Sơn", "street": "Đặng Dung"}
    },
    {
        "address": "150 guyễn Trãi, P.Hà Nam, Ninh Bình",
        "result": {"province": "Ninh Bình", "ward": "Hà Nam", "street": "Nguyễn Trãi"}
    },
    {
        "address": "88 ùi Thị Xuân, Phường Tân Phong, T Lai Châu",
        "result": {"province": "Lai Châu", "ward": "Tân Phong", "street": "Bùi Thị Xuân"}
    },

    # --- Category 2: No space after prefix ---
    {
        "address": "397 Nguyễn Khánh Toàn, PhườngHòa Cường, TP Đà Nẵng",
        "result": {"province": "Đà Nẵng", "ward": "Hòa Cường", "street": "Nguyễn Khánh Toàn"}
    },
    {
        "address": "388 Nguyễn Minh Vân, PhườngHòa Khánh, Đà Nẵng",
        "result": {"province": "Đà Nẵng", "ward": "Hòa Khánh", "street": "Nguyễn Minh Vân"}
    },
    {
        "address": "53/127 Đường Nguyễn Văn Cừ, PBồ Đề, TPHà Nội",
        "result": {"province": "Hà Nội", "ward": "Bồ Đề", "street": "Nguyễn Văn Cừ"}
    },
    {
        "address": "43 Nguyễn Thiện Kế, An Hải, TPĐà Nẵng",
        "result": {"province": "Đà Nẵng", "ward": "An Hải", "street": "Nguyễn Thiện Kế"}
    },

    # --- Category 3: Various abbreviations ---
    {
        "address": "191 Hùng Vương, P.Bắc Kạn, T.Thái Nguyên",
        "result": {"province": "Thái Nguyên", "ward": "Bắc Kạn", "street": "Hùng Vương"}
    },
    {
        "address": "63 Nguyễn Chí Thanh, Phường Tân Phong, T Lai Châu",
        "result": {"province": "Lai Châu", "ward": "Tân Phong", "street": "Nguyễn Chí Thanh"}
    },
    {
        "address": "289 Nguyễn Thông, P. Phú Thuỷ, T. Lâm Đồng",
        "result": {"province": "Lâm Đồng", "ward": "Phú Thuỷ", "street": "Nguyễn Thông"}
    },
    {
        "address": "265 Trương Định, P. Phú Lợi, Thành phố Hồ Chí Minh",
        "result": {"province": "Hồ Chí Minh", "ward": "Phú Lợi", "street": "Trương Định"}
    },

    # --- Category 4: Deep nested house numbers ---
    {
        "address": "111/138/98 Nguyễn Đức Cảnh, Thành Vinh, Nghệ An",
        "result": {"province": "Nghệ An", "ward": "Thành Vinh", "street": "Nguyễn Đức Cảnh"}
    },
    {
        "address": "68/61/135/89 Phố Nguyễn Sơn, P. Bồ Đề, TP.Hà Nội",
        "result": {"province": "Hà Nội", "ward": "Bồ Đề", "street": "Nguyễn Sơn"}
    },
    {
        "address": "79/85/144 Phố Đặng Vũ Hỷ, Phường Việt Hưng, TP Hà Nội",
        "result": {"province": "Hà Nội", "ward": "Việt Hưng", "street": "Đặng Vũ Hỷ"}
    },
    {
        "address": "93/66/52/117 Đường Ngô Gia Tự, P Việt Hưng, TP. Hà Nội",
        "result": {"province": "Hà Nội", "ward": "Việt Hưng", "street": "Ngô Gia Tự"}
    },

    # --- Category 5: "Đường số X" / "số X" ---
    {
        "address": "Đường số 8, Phường Hải Châu, Thành phố Đà Nẵng",
        "result": {"province": "Đà Nẵng", "ward": "Hải Châu", "street": "Số 8"}
    },
    {
        "address": "Đường số 95, Phường Bình Dương, Thành phố Hồ Chí Minh",
        "result": {"province": "Hồ Chí Minh", "ward": "Bình Dương", "street": "Số 95"}
    },
    {
        "address": "439 Số 6, An Hải, TP.Đà Nẵng",
        "result": {"province": "Đà Nẵng", "ward": "An Hải", "street": "Số 6"}
    },
    {
        "address": "269 số 39, P Bình Dương, TP.Hồ Chí Minh",
        "result": {"province": "Hồ Chí Minh", "ward": "Bình Dương", "street": "Số 39"}
    },

    # --- Category 6: No prefix at all ---
    {
        "address": "Nguyễn Thị Năm, Phường Bạc Liêu, Tỉnh Cà Mau",
        "result": {"province": "Cà Mau", "ward": "Bạc Liêu", "street": "Nguyễn Thị Năm"}
    },
    {
        "address": "Tây Sơn, Phường Hạc Thành, Tỉnh Thanh Hóa",
        "result": {"province": "Thanh Hóa", "ward": "Hạc Thành", "street": "Tây Sơn"}
    },
    {
        "address": "Ngô Sĩ Liên, Phường Phan Thiết, Tỉnh Lâm Đồng",
        "result": {"province": "Lâm Đồng", "ward": "Phan Thiết", "street": "Ngô Sĩ Liên"}
    },
    {
        "address": "Đinh Lễ, Phường Đồng Hới, Tỉnh Quảng Trị",
        "result": {"province": "Quảng Trị", "ward": "Đồng Hới", "street": "Đinh Lễ"}
    },

    # --- Category 7: Extra whitespace / trailing spaces ---
    {
        "address": "408 Cao Thắng, P.Ninh Kiều, Cần Thơ",
        "result": {"province": "Cần Thơ", "ward": "Ninh Kiều", "street": "Cao Thắng"}
    },
    {
        "address": " 308 Phạm Hồng Thái, P.Phố Hiến,  Hưng Yên",
        "result": {"province": "Hưng Yên", "ward": "Phố Hiến", "street": "Phạm Hồng Thái"}
    },
    {
        "address": "154 Phùng Chí Kiên, P.Bãi Cháy, Quảng Ninh",
        "result": {"province": "Quảng Ninh", "ward": "Bãi Cháy", "street": "Phùng Chí Kiên"}
    },

    # --- Category 8: Ngõ / Hẻm patterns ---
    {
        "address": "Ngõ 76 Phố Yên Phụ, Phường Tây Hồ, Thành phố Hà Nội",
        "result": {"province": "Hà Nội", "ward": "Tây Hồ", "street": "Yên Phụ"}
    },
    {
        "address": "Hẻm 50 Quang Trung, Phường Ninh Kiều, Thành phố Cần Thơ",
        "result": {"province": "Cần Thơ", "ward": "Ninh Kiều", "street": "Quang Trung"}
    },
    {
        "address": "Hẻm 103/4/2A Đường Lý Sơn, Phường Bồ Đề, Thành phố Hà Nội",
        "result": {"province": "Hà Nội", "ward": "Bồ Đề", "street": "Lý Sơn"}
    },
    {
        "address": "Ngõ 5 Phố Bùi Thiện Ngô, Phường Việt Hưng, Thành phố Hà Nội",
        "result": {"province": "Hà Nội", "ward": "Việt Hưng", "street": "Bùi Thiện Ngô"}
    },

    # --- Category 9: Very short street names ---
    {
        "address": "Đường B5, Phường Nam Nha Trang, Tỉnh Khánh Hòa",
        "result": {"province": "Khánh Hòa", "ward": "Nam Nha Trang", "street": "B5"}
    },
    {
        "address": "Đường 12, Phường Bình Dương, Thành phố Hồ Chí Minh",
        "result": {"province": "Hồ Chí Minh", "ward": "Bình Dương", "street": "12"}
    },

    # --- Category 10: Quốc lộ ---
    {
        "address": "79 Quốc lộ 4D, Phường Lào Cai, Lào Cai",
        "result": {"province": "Lào Cai", "ward": "Lào Cai", "street": "Quốc Lộ 4D"}
    },

    # --- Category 11: Ward same name as street/province ---
    {
        "address": "129/79 Đường Hồng Hà, P. Hồng Hà, Thành phố Hà Nội",
        "result": {"province": "Hà Nội", "ward": "Hồng Hà", "street": "Hồng Hà"}
    },
    {
        "address": "Phố Lê Khôi, Phường Lào Cai, Tỉnh Lào Cai",
        "result": {"province": "Lào Cai", "ward": "Lào Cai", "street": "Lê Khôi"}
    },

    # --- Category 12: Province with double spaces / missing prefix ---
    {
        "address": "53 Trần Quốc Toản, P. Phủ Lý,  Ninh Bình",
        "result": {"province": "Ninh Bình", "ward": "Phủ Lý", "street": "Trần Quốc Toản"}
    },
    {
        "address": "221 Phặc Tràng, Phong Quang, Thái Nguyên",
        "result": {"province": "Thái Nguyên", "ward": "Phong Quang", "street": "Phặc Tràng"}
    },

    # --- Category 13: 2-part addresses (only 1 comma) ---
    {
        "address": "11 Cách Mạng Tháng 8, P. Cẩm Lệ, Thành phố Đà Nẵng",
        "result": {"province": "Đà Nẵng", "ward": "Cẩm Lệ", "street": "Cách Mạng Tháng 8"}
    },

    # --- Category 14: Street name contains numbers that look like house numbers ---
    {
        "address": "Đường 2 Tháng 9, Phường Hải Châu, Thành phố Đà Nẵng",
        "result": {"province": "Đà Nẵng", "ward": "Hải Châu", "street": "2 Tháng 9"}
    },
    {
        "address": "Đường 3 Tháng 2, Phường Ninh Kiều, Thành phố Cần Thơ",
        "result": {"province": "Cần Thơ", "ward": "Ninh Kiều", "street": "3 Tháng 2"}
    },
    {
        "address": "Ba Mươi Tháng Tư, Phường Ninh Kiều, Thành phố Cần Thơ",
        "result": {"province": "Cần Thơ", "ward": "Ninh Kiều", "street": "Ba Mươi Tháng Tư"}
    },

    # --- Category 15: "An" streets that could match ward names ---
    {
        "address": "Đường An Dương Vương, Phường Ba Đình, Thành phố Hà Nội",
        "result": {"province": "Hà Nội", "ward": "Ba Đình", "street": "An Dương Vương"}
    },
    {
        "address": "An Ninh, Phường Thủy Xuân, Thành phố Huế",
        "result": {"province": "Huế", "ward": "Thủy Xuân", "street": "An Ninh"}
    },
]

# Save test cases to JSON for reuse
with open(r'v2_2_level_address\hard_test.json', 'w', encoding='utf-8') as f:
    json.dump(hard_cases, f, ensure_ascii=False, indent=2)

print(f"Created {len(hard_cases)} hard test cases")
print(f"Saved to: v2_2_level_address\\hard_test.json")
print()

# ============================================================
# Now run the solution against hard cases
# ============================================================
import re
import time
from unidecode import unidecode

# Import Solution from test_solution (reuse same class)
# Instead, we exec the solution code inline
with open(r'v2_2_level_address\public_list_province.txt', encoding='utf-8') as f:
    list_province = f.read().strip().splitlines()
with open(r'v2_2_level_address\public_list_ward.txt', encoding='utf-8') as f:
    list_ward = f.read().strip().splitlines()
with open(r'v2_2_level_address\public_list_street.txt', encoding='utf-8') as f:
    list_street = f.read().strip().splitlines()

# Import Solution from test_solution.py
sys.path.insert(0, '.')
from test_solution import Solution

solution = Solution(list_province=list_province, list_ward=list_ward, list_street=list_street)

correct = 0
total = len(hard_cases) * 3
errors = []

for i, tc in enumerate(hard_cases):
    address = tc["address"]
    expected = tc["result"]
    result = solution.process(address)

    p_ok = int(expected["province"] == result["province"])
    w_ok = int(expected["ward"] == result["ward"])
    s_ok = int(expected["street"] == result["street"])
    ok = p_ok + w_ok + s_ok
    correct += ok

    if ok < 3:
        errors.append((i, address, expected, result, p_ok, w_ok, s_ok))

score = round(correct / total * 10, 2)
print("=" * 60)
print(f"  HARD TEST RESULTS")
print(f"  Correct: {correct}/{total}")
print(f"  Score:   {score}/10")
print("=" * 60)

if errors:
    print(f"\n  Failed: {len(errors)} test cases\n")
    for idx, addr, exp, got, p, w, s in errors:
        print(f"  [{idx}] {addr}")
        if not p:
            print(f"    Province: expected '{exp['province']}' got '{got['province']}'")
        if not w:
            print(f"    Ward: expected '{exp['ward']}' got '{got['ward']}'")
        if not s:
            print(f"    Street: expected '{exp['street']}' got '{got['street']}'")
        print()
else:
    print("\n  ✓ All hard test cases passed!")
