import json

INPUT_FILE = "car_db.json"
OUTPUT_FILE = "car_db_clean.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# If the root is a list of manufacturers
for m in data:
    m.pop("example_car", None)
    m.pop("name_ocr", None)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Cleaned JSON written to", OUTPUT_FILE)
