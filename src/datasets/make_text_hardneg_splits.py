import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

RANDOM_SEED = 42
INPUT_CSV = Path("data/clinical_text_paired.csv")
OUT_DIR = Path("splits_text_hardneg")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POS_LABEL = "Effusion"
NEG_EXACT = "No Finding"

TEST_SIZE = 0.15
VAL_SIZE = 0.15

# How many abnormal non-effusion negatives to include relative to positives
HARDNEG_RATIO = 1.0  # 1.0 = same count as positives (good default)


def main():
    df = pd.read_csv(INPUT_CSV)

    # positives: contains Effusion
    pos = df[df["Finding Labels"].astype(
        str).str.contains(POS_LABEL, regex=False)].copy()
    pos["label"] = 1

    # clean normals
    neg_nf = df[df["Finding Labels"].astype(
        str).str.strip() == NEG_EXACT].copy()
    neg_nf["label"] = 0

    # abnormal non-effusion negatives (hard negatives)
    neg_abn = df[
        (df["Finding Labels"].astype(str).str.strip() != NEG_EXACT) &
        (~df["Finding Labels"].astype(str).str.contains(POS_LABEL, regex=False))
    ].copy()
    neg_abn["label"] = 0

    # sample hard negatives to match positives * ratio
    n_hard = min(len(neg_abn), int(len(pos) * HARDNEG_RATIO))
    neg_abn = neg_abn.sample(n=n_hard, random_state=RANDOM_SEED)

    task_df = pd.concat([pos, neg_nf, neg_abn], ignore_index=True)
    task_df = task_df.sample(
        frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # patient-level split
    groups = task_df["Patient ID"]

    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(gss1.split(task_df, groups=groups))
    train_val = task_df.iloc[train_val_idx].copy()
    test = task_df.iloc[test_idx].copy()

    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss2.split(
        train_val, groups=train_val["Patient ID"]))

    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    # leakage checks
    assert set(train["Patient ID"]).isdisjoint(set(val["Patient ID"]))
    assert set(train["Patient ID"]).isdisjoint(set(test["Patient ID"]))
    assert set(val["Patient ID"]).isdisjoint(set(test["Patient ID"]))

    keep = ["Image Index", "Patient ID", "Finding Labels", "text", "label"]
    train[keep].to_csv(OUT_DIR/"train.csv", index=False)
    val[keep].to_csv(OUT_DIR/"val.csv", index=False)
    test[keep].to_csv(OUT_DIR/"test.csv", index=False)

    def summarize(name, d):
        return {
            "split": name,
            "rows": len(d),
            "patients": d["Patient ID"].nunique(),
            "pos": int(d["label"].sum()),
            "neg": int((d["label"] == 0).sum()),
            "pos_rate": float(d["label"].mean()),
            "hardneg_rows_in_split": int(((d["label"] == 0) & (d["Finding Labels"] != NEG_EXACT)).sum()),
        }

    summary = pd.DataFrame(
        [summarize("train", train), summarize("val", val), summarize("test", test)])
    print("✅ Saved hard-neg text splits to:", OUT_DIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
