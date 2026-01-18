from pathlib import Path
import re
import pandas as pd
from collections import Counter, defaultdict

CORPORA = ["PO1_RUS", "PO1_ENG", "PO2_RUS", "PO3_RUS"]

# --- эвристики ---
RU_PERSON = re.compile(r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.)\b")
EN_PERSON = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.){1,2}\b")          # Vostrykh A. V.
EN_PERSON2 = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)\b")          # Steven Krug (может быть шумно)

ORG_HINTS = [
    r"университет",
    r"институт",
    r"академ",
    r"кафедр",
    r"журнал",
    r"вестн",
    r"Transactions",
    r"University",
    r"Institute",
    r"Academy",
    r"Journal",
]
ORG = re.compile(r"\b(?:[A-Z][A-Za-z&.\-]+(?:\s+[A-Z][A-Za-z&.\-]+){1,5})\b")
RU_ORG = re.compile(r"\b(?:[А-ЯЁ][А-ЯЁа-яё\"«»\-\.\s]{10,80})\b")

# города/география (минимально, чтобы не превращать в NER-проект)
GPE_WORDS = {
    "Санкт-Петербург", "Москва", "Россия", "СПб", "Пенза", "Ярославль", "Иваново",
    "Saint-Petersburg", "Russia", "Moscow"
}

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def add_hits(counter, hits, corpus, store):
    for h in hits:
        h = normalize_spaces(h)
        if len(h) < 3:
            continue
        counter[h] += 1
        store[h].add(corpus)

def classify(name: str) -> str:
    if name in GPE_WORDS:
        return "GPE"
    # простая эвристика по инициалам
    if re.search(r"\b[А-ЯЁ]\.\s*[А-ЯЁ]\.\b", name) or re.search(r"\b[A-Z]\.\s*[A-Z]\.\b", name):
        return "PERSON"
    # организации по ключевым словам
    low = name.lower()
    if any(k.lower() in low for k in ORG_HINTS):
        return "ORG"
    # по умолчанию — ORG/UNKNOWN; оставим ORG чтобы указатель был полезнее
    return "ORG"

def main():
    base = Path(".").resolve()
    out_dir = base / "out"

    total = Counter()
    seen_in = defaultdict(set)

    for corpus in CORPORA:
        p = out_dir / f"{corpus}.txt"
        if not p.exists():
            print(f"Нет файла: {p}")
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")

        # PERSON
        add_hits(total, RU_PERSON.findall(text), corpus, seen_in)
        add_hits(total, EN_PERSON.findall(text), corpus, seen_in)

        # EN_PERSON2 очень “шумный”, включим только для ENG корпуса и потом отфильтруем
        if corpus.endswith("ENG"):
            add_hits(total, EN_PERSON2.findall(text), corpus, seen_in)

        # ORG: англ. последовательности + русские длинные названия (потом чистим)
        add_hits(total, ORG.findall(text), corpus, seen_in)
        add_hits(total, RU_ORG.findall(text), corpus, seen_in)

        # GPE по словарю
        for g in GPE_WORDS:
            if g in text:
                total[g] += text.count(g)
                seen_in[g].add(corpus)

    # фильтры шума
    bad_exact = {
        "Abstract", "Keywords", "For citation", "Review article", "References",
        "In Russ", "Vol", "No", "P", "PP"
    }

    rows = []
    for name, cnt in total.items():
        if name in bad_exact:
            continue
        if cnt < 2:
            continue  # чтобы не раздувать указатель
        # убираем очевидные “общие слова” из ENG_PERSON2
        if re.fullmatch(r"[A-Z][a-z]+", name):
            continue
        rows.append({
            "name": name,
            "type": classify(name),
            "total_count": int(cnt),
            "corpora": ",".join(sorted(seen_in[name])),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("Пусто после фильтров")
        return

    df = df.sort_values(["type", "total_count", "name"], ascending=[True, False, True])
    df.to_csv(out_dir / "task3_name_index.csv", index=False, encoding="utf-8-sig")
    print("OK:", len(df), "rows")

if __name__ == "__main__":
    main()
