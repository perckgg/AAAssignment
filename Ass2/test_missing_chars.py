"""
Test missing characters in the MIDDLE or END of words (not just first char).
Simulates OCR-like corruption.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

import json
from test_solution import Solution

with open(r'v2_2_level_address\public_list_province.txt', encoding='utf-8') as f:
    list_province = f.read().strip().splitlines()
with open(r'v2_2_level_address\public_list_ward.txt', encoding='utf-8') as f:
    list_ward = f.read().strip().splitlines()
with open(r'v2_2_level_address\public_list_street.txt', encoding='utf-8') as f:
    list_street = f.read().strip().splitlines()

solution = Solution(list_province=list_province, list_ward=list_ward, list_street=list_street)

# Test cases: characters missing in MIDDLE or END
test_cases = [
    # --- Missing middle characters in STREET ---
    {"address": "100 Nguyễn Vn Cừ, Phường Bồ Đề, Thành phố Hà Nội",
     "result": {"province": "Hà Nội", "ward": "Bồ Đề", "street": "Nguyễn Văn Cừ"}},

    {"address": "50 Trần Hưg Đạo, Phường Tô Hiệu, Tỉnh Sơn La",
     "result": {"province": "Sơn La", "ward": "Tô Hiệu", "street": "Trần Hưng Đạo"}},

    {"address": "Đường Phm Viết Chánh, Phường Cẩm Lệ, Thành phố Đà Nẵng",
     "result": {"province": "Đà Nẵng", "ward": "Cẩm Lệ", "street": "Phạm Viết Chánh"}},

    {"address": "200 Lê Vn Quýnh, Phường Phong Dinh, TP. Huế",
     "result": {"province": "Huế", "ward": "Phong Dinh", "street": "Lê Văn Quýnh"}},

    {"address": "77 Nguyn Chí Thanh, Phường Tân Phong, T Lai Châu",
     "result": {"province": "Lai Châu", "ward": "Tân Phong", "street": "Nguyễn Chí Thanh"}},

    {"address": "88 Trần Quc Toản, P. Phủ Lý, Ninh Bình",
     "result": {"province": "Ninh Bình", "ward": "Phủ Lý", "street": "Trần Quốc Toản"}},

    {"address": "Đường Hoàg Hoa Thám, Phường Hà Giang 2, Tỉnh Tuyên Quang",
     "result": {"province": "Tuyên Quang", "ward": "Hà Giang 2", "street": "Hoàng Hoa Thám"}},

    {"address": "120 Ngô Quyn, P.An Hội, Vĩnh Long",
     "result": {"province": "Vĩnh Long", "ward": "An Hội", "street": "Ngô Quyền"}},

    # --- Missing last character in STREET ---
    {"address": "100 Nguyễn Văn C, Phường Bồ Đề, Thành phố Hà Nội",
     "result": {"province": "Hà Nội", "ward": "Bồ Đề", "street": "Nguyễn Văn Cừ"}},

    {"address": "Đường Phạm Viết Chán, Phường Cẩm Lệ, Đà Nẵng",
     "result": {"province": "Đà Nẵng", "ward": "Cẩm Lệ", "street": "Phạm Viết Chánh"}},

    {"address": "55 Cao Thắn, P.Ninh Kiều, Cần Thơ",
     "result": {"province": "Cần Thơ", "ward": "Ninh Kiều", "street": "Cao Thắng"}},

    # --- Missing middle characters in WARD ---
    {"address": "100 Cao Thắng, P.Nin Kiều, Cần Thơ",
     "result": {"province": "Cần Thơ", "ward": "Ninh Kiều", "street": "Cao Thắng"}},

    {"address": "200 Hùng Vương, P.Bắ Kạn, T.Thái Nguyên",
     "result": {"province": "Thái Nguyên", "ward": "Bắc Kạn", "street": "Hùng Vương"}},

    {"address": "150 Đinh Lễ, Phường Đồg Hới, Tỉnh Quảng Trị",
     "result": {"province": "Quảng Trị", "ward": "Đồng Hới", "street": "Đinh Lễ"}},

    # --- Missing characters in PROVINCE ---
    {"address": "100 Cao Thắng, P.Ninh Kiều, Cầ Thơ",
     "result": {"province": "Cần Thơ", "ward": "Ninh Kiều", "street": "Cao Thắng"}},

    {"address": "50 Hùng Vương, P.Bắc Kạn, Thái Nguên",
     "result": {"province": "Thái Nguyên", "ward": "Bắc Kạn", "street": "Hùng Vương"}},

    {"address": "80 Đinh Lễ, Phường Đồng Hới, Quảg Trị",
     "result": {"province": "Quảng Trị", "ward": "Đồng Hới", "street": "Đinh Lễ"}},

    {"address": "300 Trương Định, P. Phú Lợi, Hồ Chí Min",
     "result": {"province": "Hồ Chí Minh", "ward": "Phú Lợi", "street": "Trương Định"}},

    # --- Multiple missing characters (severe corruption) ---
    {"address": "99 Nguyễ Thiện Kế, An Hải, Đà Nẵg",
     "result": {"province": "Đà Nẵng", "ward": "An Hải", "street": "Nguyễn Thiện Kế"}},

    {"address": "Đường Cù Chín Lan, Phường Quang Trun, Tỉnh Thanh Hóa",
     "result": {"province": "Thanh Hóa", "ward": "Quang Trung", "street": "Cù Chính Lan"}},
]

print(f"Testing {len(test_cases)} cases with missing MIDDLE/END characters...")
print()

correct = 0
total = len(test_cases) * 3
errors = []

for i, tc in enumerate(test_cases):
    result = solution.process(tc["address"])
    expected = tc["result"]
    p_ok = int(expected["province"] == result["province"])
    w_ok = int(expected["ward"] == result["ward"])
    s_ok = int(expected["street"] == result["street"])
    ok = p_ok + w_ok + s_ok
    correct += ok
    if ok < 3:
        errors.append((i, tc["address"], expected, result, p_ok, w_ok, s_ok))

print("=" * 60)
print(f"  Result: {correct}/{total} = {round(correct/total*10, 2)}/10")
print("=" * 60)

if errors:
    print(f"\n  Failed: {len(errors)} / {len(test_cases)} test cases\n")
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
    print("\n  ✓ All test cases passed!")
