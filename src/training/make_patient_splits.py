import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# -------------------------
# CONFIG
# -------------------------
RANDOM_SEED = 42

# Input: the file you generated in Phase 1
INPUT_CSV = Path("data/clinical_text_paired.csv")

# Output folder
OUT_DIR = Path("splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Task definition (Option A: clean negatives)
POS_LABEL_STR = "Effusion"
NEG_EXACT_STR = "No Finding"

# Split ratios
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # as fraction of total dataset (handled in 2-step split)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_CSV}. Run Phase 1 first to generate it.")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["Image Index", "Patient ID",
                     "Finding Labels", "text", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")

    # -------------------------
    # FILTER TO CLEAN BINARY TASK
    # Positives: contains Effusion
    # Negatives: exactly No Finding
    # -------------------------
    pos = df[df["Finding Labels"].astype(str).str.contains(
        POS_LABEL_STR, regex=False)].copy()
    neg = df[df["Finding Labels"].astype(
        str).str.strip() == NEG_EXACT_STR].copy()

    # Set labels explicitly for safety
    pos["label"] = 1
    neg["label"] = 0

    task_df = pd.concat([pos, neg], ignore_index=True)

    # Shuffle rows (deterministic)
    task_df = task_df.sample(
        frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # -------------------------
    # PATIENT-LEVEL SPLIT
    # Step 1: train_val vs test
    # Step 2: train vs val (from train_val)
    # -------------------------
    groups = task_df["Patient ID"]

    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(gss1.split(task_df, groups=groups))

    train_val = task_df.iloc[train_val_idx].copy()
    test = task_df.iloc[test_idx].copy()

    # Now split train_val into train and val.
    # We want VAL_SIZE of total, so val_fraction_of_trainval = VAL_SIZE / (1 - TEST_SIZE)
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)

    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss2.split(
        train_val, groups=train_val["Patient ID"]))

    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    # -------------------------
    # SANITY CHECK: no patient overlap
    # -------------------------
    train_p = set(train["Patient ID"])
    val_p = set(val["Patient ID"])
    test_p = set(test["Patient ID"])

    assert train_p.isdisjoint(val_p), "Leakage: train/val share patients!"
    assert train_p.isdisjoint(test_p), "Leakage: train/test share patients!"
    assert val_p.isdisjoint(test_p), "Leakage: val/test share patients!"

    # -------------------------
    # SAVE
    # -------------------------
    keep_cols = ["Image Index", "Patient ID",
                 "Finding Labels", "text", "label"]

    train[keep_cols].to_csv(OUT_DIR / "train.csv", index=False)
    val[keep_cols].to_csv(OUT_DIR / "val.csv", index=False)
    test[keep_cols].to_csv(OUT_DIR / "test.csv", index=False)

    # -------------------------
    # PRINT SUMMARY
    # -------------------------
    def summarize(name, split_df):
        return {
            "split": name,
            "rows": len(split_df),
            "patients": split_df["Patient ID"].nunique(),
            "pos": int(split_df["label"].sum()),
            "neg": int((split_df["label"] == 0).sum()),
            "pos_rate": float(split_df["label"].mean()),
        }

    summary = pd.DataFrame([
        summarize("train", train),
        summarize("val", val),
        summarize("test", test),
    ])

    print("\n✅ Patient-level splits saved to ./splits/")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
