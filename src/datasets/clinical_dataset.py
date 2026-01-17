import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = "nih_sample/sample/sample_labels.csv"     # or your path
OUT_CSV = "data/clinical_text_paired.csv"

RANDOM_SEED = 42

POS_LABEL = "Effusion"
NEG_EXACT = "No Finding"

# Controls to reduce label leakage:
P_EXPLICIT_POS = 0.25   # only 25% of positive reports explicitly say "pleural effusion"
# only 35% of normals explicitly say "No pleural effusion"
P_NO_PLEURAL_EFFUSION_IN_NORMAL = 0.35

# Optional: inject small label noise into text (NOT into labels!)
# This means sometimes a positive report sounds less clear, and sometimes normal is less clean.
P_TEXT_AMBIGUITY = 0.20

# Optional: corruption experiment (set >0 for robustness). Keep 0.0 for main dataset.
P_CORRUPT_TEXT = 0.0     # e.g., 0.15 to randomly drop a sentence 15% of the time

# -----------------------------
# RNG
# -----------------------------
RNG = np.random.default_rng(RANDOM_SEED)


def choose(xs):
    return xs[int(RNG.integers(0, len(xs)))]


def maybe(p: float) -> bool:
    return bool(RNG.random() < p)


def corrupt_text(sentences):
    """Randomly drop one sentence to simulate incomplete documentation."""
    if len(sentences) <= 1:
        return sentences
    drop_idx = int(RNG.integers(0, len(sentences)))
    return [s for i, s in enumerate(sentences) if i != drop_idx]


def generate_report_from_labels(labels: str) -> str:
    labels = str(labels).strip()

    # Banks (avoid laterality)
    pos_explicit = [
        "Small pleural effusion is present.",
        "Pleural effusion is noted.",
        "Findings are consistent with a pleural effusion."
    ]
    pos_implicit = [
        "Blunting of the costophrenic angle is noted.",
        "There is dependent basilar opacity which may reflect layering fluid.",
        "Mild fluid layering is suspected.",
        "Subtle costophrenic angle blunting suggests a small volume of fluid.",
        "A small dependent opacity pattern is present, which can be seen with pleural fluid."
    ]
    pos_ambiguous = [
        "A subtle basilar opacity is present; correlate clinically.",
        "Basilar changes are present; etiology is nonspecific.",
        "Mild basilar opacity is present; follow-up imaging may be helpful."
    ]

    normal_findings = [
        "The lungs are clear.",
        "No acute cardiopulmonary abnormality is identified.",
        "Cardiomediastinal silhouette is within normal limits.",
        "No focal airspace disease is seen.",
        "No acute intrathoracic process is identified."
    ]
    normal_impression = [
        "No acute disease.",
        "No acute cardiopulmonary process.",
        "Normal chest radiograph."
    ]

    # Negations (non-lateral)
    neg_common = [
        "No pneumothorax.",
        "No focal consolidation.",
        "No cardiomegaly.",
        "No acute osseous abnormality."
    ]
    # Keep this optional to avoid making normals perfectly separable
    neg_pleural = "No pleural effusion."

    uncertainty = [
        "Clinical correlation is recommended.",
        "Comparison with prior imaging would be helpful.",
        "Assessment is limited by low lung volumes.",
        "Technique is suboptimal; correlate clinically."
    ]

    sentences = []
    use_headers = maybe(0.6)

    is_pos = (POS_LABEL in labels)
    is_clean_normal = (labels == NEG_EXACT)

    findings = []
    impression = []

    if is_pos:
        # Implicit most of the time, explicit sometimes
        if maybe(1.0 - P_TEXT_AMBIGUITY):
            findings.append(choose(pos_implicit))
        else:
            findings.append(choose(pos_ambiguous))

        if maybe(P_EXPLICIT_POS) and (not maybe(P_TEXT_AMBIGUITY)):
            impression.append(choose(pos_explicit))
        else:
            # softer wording
            impression.append(choose([
                "Findings may reflect a small pleural fluid volume.",
                "A small pleural fluid component is suspected.",
                "Consider pleural fluid; correlate clinically."
            ]))

        if maybe(0.75):
            findings.append(choose(neg_common))

        if maybe(0.25):
            impression.append(choose(uncertainty))

    elif is_clean_normal:
        findings.append(choose(normal_findings))

        # Add some negations, but not always all of them
        if maybe(0.70):
            findings.append(choose(neg_common))

        # Only sometimes include the “No pleural effusion” phrase
        if maybe(P_NO_PLEURAL_EFFUSION_IN_NORMAL) and (not maybe(P_TEXT_AMBIGUITY)):
            findings.append(neg_pleural)

        # Sometimes add mild uncertainty (to prevent trivially separable wording)
        if maybe(P_TEXT_AMBIGUITY):
            impression.append(choose([
                "No acute abnormality is identified; correlate clinically.",
                "No acute process; clinical correlation is advised."
            ]))
        else:
            impression.append(choose(normal_impression))

        if maybe(0.15):
            impression.append(choose(uncertainty))

    else:
        # Other abnormalities: generic abnormal description (kept for full CSV pairing)
        findings.append(
            f"Reported findings include: {labels.replace('|', ', ').lower()}.")
        impression.append("Clinical correlation is recommended.")

    # Compose sentences
    if use_headers:
        sentences.append(f"FINDINGS: {' '.join(findings)}")
        sentences.append(f"IMPRESSION: {' '.join(impression)}")
    else:
        sentences.extend(findings + impression)

    # Optional corruption experiment
    if P_CORRUPT_TEXT > 0 and maybe(P_CORRUPT_TEXT):
        sentences = corrupt_text(sentences)

    return " ".join(" ".join(sentences).split())


def main():
    df = pd.read_csv(INPUT_CSV)

    required = ["Image Index", "Patient ID", "Finding Labels"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")

    # Label stays exactly as before (ground truth)
    df["label"] = df["Finding Labels"].astype(
        str).str.contains(POS_LABEL, regex=False).astype(int)

    # New text generation (harder / less leaky)
    df["text"] = df["Finding Labels"].astype(
        str).apply(generate_report_from_labels)

    out = df[["Image Index", "Patient ID",
              "Finding Labels", "text", "label"]].copy()
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # Print quick stats
    pos = int(out["label"].sum())
    neg = int((out["label"] == 0).sum())
    print("Saved:", OUT_CSV)
    print("Rows:", len(out))
    print("Pos (Effusion):", pos)
    print("Neg:", neg)

    # Show examples
    ex_pos = out[out["label"] == 1].sample(3, random_state=1)[
        ["Finding Labels", "text"]]
    ex_neg = out[out["label"] == 0].sample(3, random_state=1)[
        ["Finding Labels", "text"]]
    print("\n--- Example positives ---")
    for _, r in ex_pos.iterrows():
        print("-", r["text"])
    print("\n--- Example negatives ---")
    for _, r in ex_neg.iterrows():
        print("-", r["text"])


if __name__ == "__main__":
    main()
