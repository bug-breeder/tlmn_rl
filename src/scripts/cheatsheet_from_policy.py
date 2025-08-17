import argparse, pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="dataset.csv")
    p.add_argument("--out", type=str, default="cheatsheet.md")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    sections = []

    def section(title, cond):
        sub = df.query(cond) if cond else df
        if len(sub)==0:
            sections.append(f"### {title}\nKhông đủ dữ liệu.\n")
            return
        top = sub["action_family"].value_counts(normalize=True).head(5)
        sections.append(f"### {title}")
        for fam, p in top.items():
            sections.append(f"- **{fam}** ~ {p:.1%}")
        sections.append("")

    section("MỞ LƯỢT – tay ngắn (≤5)", "last_kind=='none' and hand_size<=5")
    section("MỞ LƯỢT – tay trung (6–9)", "last_kind=='none' and hand_size>=6 and hand_size<=9")
    section("MỞ LƯỢT – tay dài (≥10)", "last_kind=='none' and hand_size>=10")
    section("ĐUỔI – đơn", "last_kind=='single'")
    section("ĐUỔI – đôi", "last_kind=='pair'")
    section("ĐUỔI – sảnh", "last_kind=='straight'")
    section("ĐUỔI – đôi thông", "last_kind=='pair_run'")
    section("GẶP HE0 (2) – đuổi đơn", "last_kind=='single' and last_is_2==True")
    section("GẶP ĐÔI HE0 (2) – đuổi đôi", "last_kind=='pair' and last_is_2==True")

    Path(args.out).write_text("\n".join(sections), encoding="utf-8")
    print(f"Saved cheat sheet to {args.out}")

if __name__ == "__main__":
    main()
