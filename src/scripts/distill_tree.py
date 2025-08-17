import argparse, pandas as pd, numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="dataset.csv")
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--min_samples_leaf", type=int, default=200)
    p.add_argument("--out_rules", type=str, default="rules.txt")
    p.add_argument("--out_md", type=str, default="cheatsheet.md")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    # Encode categorical
    cat_cols = ["hand_bucket","last_kind"]
    for c in cat_cols:
        df[c] = df[c].astype("category")
        df[c] = df[c].cat.codes

    y = df["action_family"].astype("category").cat.codes
    X = df.drop(columns=["action_family"])
    X = X.fillna(0)

    clf = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, random_state=0, class_weight="balanced")
    clf.fit(X, y)

    rules = export_text(clf, feature_names=list(X.columns))
    Path(args.out_rules).write_text(rules, encoding="utf-8")

    # Simple cheatsheet using frequency aggregations
    def topk(sub, k=3):
        vc = sub["action_family"].value_counts(normalize=True)
        return [(i, float(p)) for i,p in vc.head(k).items()]

    def section(df, title, query=None):
        if query is not None:
            sub = df.query(query)
        else:
            sub = df
        lines = [f"### {title}"]
        if len(sub)==0:
            lines.append("Không đủ dữ liệu.")
            return "\n".join(lines)
        for fam, p in topk(sub, 5):
            lines.append(f"- **{fam}** ~ {p:.1%}")
        return "\n".join(lines)

    parts = []
    parts.append("# Cheat‑sheet rút từ policy\n")
    parts.append(section(df, "MỞ LƯỢT – tay ngắn (≤5)", "last_kind=='none' and hand_size<=5"))
    parts.append(section(df, "MỞ LƯỢT – tay trung (6–9)", "last_kind=='none' and hand_size>=6 and hand_size<=9"))
    parts.append(section(df, "MỞ LƯỢT – tay dài (≥10)", "last_kind=='none' and hand_size>=10"))
    parts.append(section(df, "ĐUỔI – đơn", "last_kind=='single'"))
    parts.append(section(df, "ĐUỔI – đôi", "last_kind=='pair'"))
    parts.append(section(df, "ĐUỔI – sảnh", "last_kind=='straight'"))
    parts.append(section(df, "ĐUỔI – đôi thông", "last_kind=='pair_run'"))
    parts.append(section(df, "ĐỐI MẶT HE0 (2) – đơn", "last_kind=='single' and last_is_2==True"))
    parts.append(section(df, "ĐỐI MẶT ĐÔI HE0 (2)", "last_kind=='pair' and last_is_2==True"))
    Path(args.out_md).write_text("\n\n".join(parts), encoding="utf-8")

    print(f"Saved distilled rules to {args.out_rules} and cheat sheet to {args.out_md}")

if __name__ == "__main__":
    main()
