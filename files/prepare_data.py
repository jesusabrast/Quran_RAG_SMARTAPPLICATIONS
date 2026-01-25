import pandas as pd
from pathlib import Path


def load_quran_file(filepath: Path):
    """Load a simple pipe-delimited quran file (ref|aya|text...) into a dict keyed by 'ref|aya'."""
    data = {}
    if not filepath.exists():
        print(f"Warning: file not found: {filepath}")
        return data
    with filepath.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 3:
                continue
            key = f"{parts[0]}|{parts[1]}"
            text = '|'.join(parts[2:])
            data[key] = text
    return data


print("Loading Arabic and English source files...")
HERE = Path(__file__).resolve().parent
arabic_path = HERE / 'quran-simple.txt'
english_path = HERE / 'en.maududi.txt'

arabic_text = load_quran_file(arabic_path)
english_text = load_quran_file(english_path)

if not arabic_text:
    print(f"Error: Arabic source not found or empty: {arabic_path}")
    raise SystemExit(1)

if not english_text:
    print(f"Warning: English translation not found: {english_path}. English column will be empty.")

prepared = []
print("Creating bilingual dataset (Arabic + English)...")
for key, a_text in arabic_text.items():
    prepared.append({
        "reference": key,
        "arabic": a_text,
        "translation_english": english_text.get(key, "")
    })

df = pd.DataFrame(prepared)

# Write the bilingual CSV to the repository root so the app can load it reliably
OUT = HERE.parent / 'quran_bilingual_data.csv'
df.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f"Success: wrote bilingual CSV -> {OUT}")