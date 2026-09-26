from src.preprocessing import normalize_name, normalize_address


name_tests = [
    "ABC CORPORATION",
    "ABC Corporation",
    "-- Shivshakti Pvt. Ltd. | www.shivshakti.com",
    "LLC Moncada Léarning Center",
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "Jones, Martinez & Mcintosh Corp [INCORPORATED]",
]


print("=" * 70)
print("NAME NORMALIZATION")
print("=" * 70)

for name in name_tests:
    result = normalize_name(name)

    print(f"\nRAW:        {result['original']}")
    print(f"NORMALIZED: {result['normalized']}")
    print(f"CORE:       {result['core']}")


address_tests = [
    "19 1/2 STARDUST TRAIL, GREENSBORO, NC",
    "GREENSBORO, NC, 19 1/2 STARDUST TRAIL",
    "19320 1st Place, Washington, DC",
    "19320- 1ND PL, Washington, DC",
]


print("\n")
print("=" * 70)
print("ADDRESS NORMALIZATION")
print("=" * 70)

for address in address_tests:
    result = normalize_address(address)

    print(f"\nRAW:        {result['original']}")
    print(f"NORMALIZED: {result['normalized']}")
    print(f"NUMBERS:    {result['number_tokens']}")