import re
from collections import Counter
from pathlib import Path

import fitz  # pymupdf
import pandas as pd

# --- настройки ---
PDF_FILES = [
    "PO1_RUS.pdf",
    "PO1_ENG.pdf",
    "PO2_RUS.pdf",
    "PO3_RUS.pdf",
]

# минимальные стоп-слова (можно расширять)
BASIC_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as", "by", "at", "from",
    "this", "that", "these", "those",
    "это", "как", "так", "и", "или", "в", "во", "на", "по", "к", "из", "для", "с", "со", "о", "об", "от",
    "этот", "эта", "эти", "тот", "та", "те",
}

# аббревиатуры будем выделять отдельно и исключать из частотника
EXCLUDE_ABBREVIATIONS_FROM_FREQ = True
ABBR_MIN_LEN = 2
ABBR_MAX_LEN = 12

def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts)

def tokenize_ru(text: str):
    text = text.lower()
    tokens = re.findall(r"[а-яё]+(?:-[а-яё]+)?", text)
    tokens = [t for t in tokens if len(t) >= 3 and t not in BASIC_STOPWORDS]
    return tokens

def tokenize_en(text: str):
    text = text.lower()
    tokens = re.findall(r"[a-z]+(?:-[a-z]+)?", text)
    tokens = [t for t in tokens if len(t) >= 3 and t not in BASIC_STOPWORDS]
    return tokens

def extract_abbreviations(text: str):
    """
    Достаём аббревиатуры из исходного текста (в верхнем регистре), включая цифры:
    SCPS, HDT, PCSK9, LDL, ORION-10 (ORION будет поймано, "10" отдельно не надо).
    """
    abbrs = re.findall(rf"\b[А-ЯЁA-Z0-9]{{{ABBR_MIN_LEN},{ABBR_MAX_LEN}}}\b", text)
    # убираем чисто цифровые "2024" и т.п.
    abbrs = [a for a in abbrs if not a.isdigit()]
    # уникальные + сортировка
    return sorted(set(abbrs))

def make_frequency_df(tokens):
    c = Counter(tokens)
    df = pd.DataFrame(c.items(), columns=["word", "freq"]).sort_values("freq", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df

def calc_stats(tokens, df):
    N = len(tokens)               # число словоупотреблений
    V = int(df.shape[0])          # число разных слов
    hapax = int((df["freq"] == 1).sum())
    top_word = df.iloc[0]["word"] if V else ""
    top_freq = int(df.iloc[0]["freq"]) if V else 0
    return {
        "N_tokens": N,
        "V_types": V,
        "hapax_1freq": hapax,
        "top_word": top_word,
        "top_freq": top_freq,
    }

def main():
    base = Path(".").resolve()
    out_dir = base / "out"
    out_dir.mkdir(exist_ok=True)

    summary_rows = []

    for name in PDF_FILES:
        pdf_path = base / name
        if not pdf_path.exists():
            print(f"Нет файла: {pdf_path}")
            continue

        text = extract_text_from_pdf(pdf_path)

        # 1) сохраняем извлечённый текст
        (out_dir / f"{pdf_path.stem}.txt").write_text(text, encoding="utf-8")

        # 2) аббревиатуры отдельным файлом (для предметного указателя/раздела "Аббревиатуры")
        abbrs = extract_abbreviations(text)
        pd.DataFrame({"abbr": abbrs}).to_csv(
            out_dir / f"{pdf_path.stem}_abbr.csv",
            index=False,
            encoding="utf-8-sig"
        )

        # 3) токенизация по языку
        if pdf_path.stem.endswith("_ENG"):
            tokens = tokenize_en(text)
        else:
            tokens = tokenize_ru(text)

        # 4) исключаем аббревиатуры из частотного словника (но аббревиатуры уже сохранены отдельно)
        if EXCLUDE_ABBREVIATIONS_FROM_FREQ and abbrs:
            abbrs_lower = {a.lower() for a in abbrs}
            tokens = [t for t in tokens if t not in abbrs_lower]

        # 5) частотный словник
        df = make_frequency_df(tokens)
        df.to_csv(out_dir / f"{pdf_path.stem}_freq.csv", index=False, encoding="utf-8-sig")

        # 6) сводные показатели
        stats = calc_stats(tokens, df)
        stats["corpus"] = pdf_path.stem
        summary_rows.append(stats)

        print(f"OK: {name} | tokens={stats['N_tokens']} | types={stats['V_types']} | top={stats['top_word']}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)[
            ["corpus", "N_tokens", "V_types", "hapax_1freq", "top_word", "top_freq"]
        ]
        summary_df.to_csv(out_dir / "task1_summary.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
