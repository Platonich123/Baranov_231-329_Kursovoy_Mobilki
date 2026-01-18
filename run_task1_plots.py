from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CORPORA = ["PO1_RUS", "PO1_ENG", "PO2_RUS", "PO3_RUS"]
TOP_N = 30

def plot_top_bar(df, title, out_path: Path, top_n=30):
    top = df.sort_values("freq", ascending=False).head(top_n).copy()
    top = top.sort_values("freq", ascending=True)  # чтобы красивые горизонтальные бары

    plt.figure(figsize=(10, 6))
    plt.barh(top["word"], top["freq"])
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_step(rank, freq, title, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.step(rank, freq, where="post")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_loglog(rank, freq, title, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.loglog(rank, freq, "o-", markersize=3, linewidth=1)
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    base = Path(".").resolve()
    out_dir = base / "out"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for corpus in CORPORA:
        freq_csv = out_dir / f"{corpus}_freq.csv"
        if not freq_csv.exists():
            print(f"Нет файла: {freq_csv}")
            continue

        df = pd.read_csv(freq_csv, encoding="utf-8-sig")

        # 1) гистограмма топ-N
        plot_top_bar(
            df,
            title=f"{corpus}: top-{TOP_N} word frequencies",
            out_path=plots_dir / f"{corpus}_top{TOP_N}_bar.png",
            top_n=TOP_N,
        )

        # для rank-графиков сортируем по рангу
        df_ranked = df.sort_values("rank")
        rank = df_ranked["rank"].tolist()
        freq = df_ranked["freq"].tolist()

        # 2) step
        plot_step(
            rank, freq,
            title=f"{corpus}: rank–frequency (step)",
            out_path=plots_dir / f"{corpus}_rank_freq_step.png",
        )

        # 3) log-log
        plot_loglog(
            rank, freq,
            title=f"{corpus}: rank–frequency (log-log)",
            out_path=plots_dir / f"{corpus}_rank_freq_loglog.png",
        )

        print(f"OK plots: {corpus}")

if __name__ == "__main__":
    main()
