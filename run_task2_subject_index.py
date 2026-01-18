from pathlib import Path
import re
import pandas as pd

CORPORA = ["PO1_RUS", "PO1_ENG", "PO2_RUS", "PO3_RUS"]
TARGET_TERMS = 100

# Слова, которые НЕ должны попадать в предметный указатель
RU_STOP = {
    "что","это","как","так","и","или","в","во","на","по","к","из","для","с","со","о","об","от",
    "при","быть","может","могут","можно","которые","которых","которого","которыми",
    "также","однако","где","когда","если","то","его","ее","их","себя","собой","нам","вам","они",
    "время","годы","год","данном","данные","число","например","вполне","вряд","между",
    "более","менее","сам","сама","сами","этот","эта","эти","тот","та","те",
    # мусор/оформление
    "рис","таблица","таб","удк","url","doi",
}

EN_STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","as","by","at","from",
    "this","that","these","those","is","are","was","were","be","been","being",
    "it","its","they","their","them","we","our","you","your","i",
    # мусор/оформление
    "doi","url","fig","table","tables",
    # часто “формульное”
    "min","max","inf",
}

def normalize_abbr(a: str) -> str:
    a = a.strip()
    # приведение к верхнему регистру + латиница/кириллица как есть
    return a.upper()

def is_russian_word(w: str) -> bool:
    return bool(re.fullmatch(r"[а-яё]+(?:-[а-яё]+)?", w, flags=re.IGNORECASE))

def is_english_word(w: str) -> bool:
    return bool(re.fullmatch(r"[a-z]+(?:-[a-z]+)?", w, flags=re.IGNORECASE))

def load_freq(out_dir: Path, corpus: str) -> pd.DataFrame:
    p = out_dir / f"{corpus}_freq.csv"
    df = pd.read_csv(p, encoding="utf-8-sig")
    df["corpus"] = corpus
    return df

def load_abbr(out_dir: Path, corpus: str) -> pd.DataFrame:
    p = out_dir / f"{corpus}_abbr.csv"
    df = pd.read_csv(p, encoding="utf-8-sig")
    df["abbr"] = df["abbr"].astype(str).map(normalize_abbr)
    df["corpus"] = corpus
    return df

def dedup_abbreviations(abbrs):
    """
    Склеиваем “парные” варианты латиница/кириллица, которые часто встречаются:
    SCPS/СКФС, LETI/ЛЭТИ, EM/ЕМ.
    Возвращаем список строк для Word.
    """
    s = set(abbrs)

    pairs = [
        ("SCPS", "СКФС"),
        ("LETI", "ЛЭТИ"),
        ("EM", "ЕМ"),
    ]

    merged = []
    used = set()
    for lat, cyr in pairs:
        if lat in s or cyr in s:
            if lat in s and cyr in s:
                merged.append(f"{lat} ({cyr})")
                used.add(lat); used.add(cyr)
            elif lat in s:
                merged.append(lat); used.add(lat)
            else:
                merged.append(cyr); used.add(cyr)

    # остальное (что не склеили)
    rest = sorted([a for a in s if a not in used])
    return merged + rest

def main():
    base = Path(".").resolve()
    out_dir = base / "out"
    out_dir.mkdir(exist_ok=True)

    # --- 1) Аббревиатуры (общий список) ---
    abbr_frames = []
    for c in CORPORA:
        p = out_dir / f"{c}_abbr.csv"
        if p.exists():
            abbr_frames.append(load_abbr(out_dir, c))
        else:
            print(f"Нет abbr: {p}")

    if abbr_frames:
        abbr_all = pd.concat(abbr_frames, ignore_index=True)
        abbr_unique = sorted(set(abbr_all["abbr"].tolist()))
        abbr_for_word = dedup_abbreviations(abbr_unique)

        abbr_out = pd.DataFrame({"abbr": abbr_for_word})
        abbr_out.to_csv(out_dir / "task2_abbreviations.csv", index=False, encoding="utf-8-sig")
    else:
        print("Не найдено ни одного *_abbr.csv")

    # --- 2) Предметный указатель (100 понятий) ---
    freq_frames = []
    for c in CORPORA:
        p = out_dir / f"{c}_freq.csv"
        if p.exists():
            freq_frames.append(load_freq(out_dir, c))
        else:
            print(f"Нет freq: {p}")

    if not freq_frames:
        print("Не найдено ни одного *_freq.csv")
        return

    freq_all = pd.concat(freq_frames, ignore_index=True)

    # суммируем частоты по слову по всем корпусам
    agg = freq_all.groupby("word", as_index=False)["freq"].sum()
    agg["word"] = agg["word"].astype(str)

    # фильтрация: убираем мусор/служебные слова
    cleaned = []
    for w, f in zip(agg["word"].tolist(), agg["freq"].tolist()):
        wl = w.lower().strip()
        if not wl or len(wl) < 3:
            continue

        if is_russian_word(wl):
            if wl in RU_STOP:
                continue
        elif is_english_word(wl):
            if wl in EN_STOP:
                continue
        else:
            # не слово (смешанные символы) — не берём в указатель
            continue

        cleaned.append((wl, int(f)))

    cleaned_df = pd.DataFrame(cleaned, columns=["term", "total_freq"])
    cleaned_df = cleaned_df.sort_values(["total_freq", "term"], ascending=[False, True])

    # берём TOP-100
    subject_df = cleaned_df.head(TARGET_TERMS).copy()
    subject_df["rank"] = range(1, len(subject_df) + 1)

    subject_df.to_csv(out_dir / "task2_subject_index.csv", index=False, encoding="utf-8-sig")

    print(f"OK: subject index terms = {len(subject_df)}")
    if abbr_frames:
        print(f"OK: abbreviations = {len(abbr_for_word)}")

if __name__ == "__main__":
    main()
