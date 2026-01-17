# Synthetic Clinical Text Generator (Effusion vs No Finding)
# Produces: data/clinical_text_paired.csv
# Input: nih_sample/sample/sample_labels.csv (or your sample_labels.csv)

import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = "nih_sample/sample/sample_labels.csv"   # change if needed
OUT_CSV = "data/clinical_text_paired.csv"
RANDOM_SEED = 42

POS_LABEL = "Effusion"      # target task: Effusion vs No Finding
NEG_LABEL = "No Finding"    # used later for "clean negatives" if you choose

# -----------------------------
# RNG (reproducible)
# -----------------------------
RNG = np.random.default_rng(RANDOM_SEED)


def choose(xs):
    return xs[int(RNG.integers(0, len(xs)))]


def maybe(p: float) -> bool:
    return bool(RNG.random() < p)


def generate_synthetic_report(finding_labels: str) -> str:
    """
    Generate a radiology-style snippet paired to the label string.
    - Variable templates
    - Negations
    - Optional uncertainty
    - No laterality (no left/right)
    - Anti-cheat: positives often use implicit wording (not always 'pleural effusion')
    """
    labels = str(finding_labels)

    use_headers = maybe(0.6)

    # --- Phrase banks ---
    effusion_explicit = [
        "Small pleural effusion is present.",
        "Pleural effusion is noted.",
        "Findings are consistent with pleural effusion."
    ]
    effusion_implicit = [
        "Blunting of the costophrenic angle suggests pleural fluid.",
        "There is basilar opacity with a configuration suggesting layering fluid.",
        "Dependent fluid is suspected within the pleural space.",
        "Mild fluid accumulation is suspected.",
        "A small amount of pleural fluid is suspected."
    ]

    normal_findings = [
        "The lungs are clear.",
        "No acute cardiopulmonary abnormality is identified.",
        "Cardiomediastinal silhouette is within normal limits.",
        "No focal airspace disease is seen.",
        "No acute intrathoracic process is identified."
    ]

    # Negations (safe, non-lateral)
    negations_common = [
        "No pneumothorax.",
        "No focal consolidation.",
        "No cardiomegaly.",
        "No acute osseous abnormality."
    ]
    # Extra negatives commonly listed on normal exams
    negations_normal_extra = [
        "No pleural effusion.",
        "No edema."
    ]

    uncertainty = [
        "Clinical correlation is recommended.",
        "Comparison with prior imaging would be helpful.",
        "Assessment is limited by low lung volumes.",
        "Technique is suboptimal; correlate clinically."
    ]

    findings = []
    impression = []

    if POS_LABEL in labels:
        # Mix implicit/explicit to reduce pure keyword detection
        if maybe(0.70):
            findings.append(choose(effusion_implicit))
        if maybe(0.55):
            impression.append(choose(effusion_explicit))
        else:
            impression.append("Findings suggest pleural fluid.")

        # Add relevant negatives (avoid contradiction: do NOT add "No pleural effusion" here)
        if maybe(0.80):
            findings.append(choose(negations_common))

        # Optional additional statement
        if maybe(0.35):
            findings.append(choose([
                "No significant mediastinal widening.",
                "No large pleural collection.",
                "No displaced rib fracture is seen."
            ]))

    elif labels.strip() == NEG_LABEL:
        findings.append(choose(normal_findings))

        # Normal reports often list multiple negatives, including "No pleural effusion"
        if maybe(0.85):
            pool = negations_common + negations_normal_extra
            k = int(RNG.integers(2, 4))  # choose 2-3
            chosen = RNG.choice(pool, size=k, replace=False)
            findings.extend(list(chosen))

        impression.append(choose([
            "No acute disease.",
            "Normal chest radiograph.",
            "No acute cardiopulmonary process."
        ]))

    else:
        # For non-effusion abnormal cases (kept for full-dataset pairing)
        findings.append(
            f"Reported findings include: {labels.replace('|', ', ').lower()}.")
        impression.append("Clinical correlation is recommended.")

    # Optional uncertainty/noise suffix
    if maybe(0.25):
        impression.append(choose(uncertainty))

    # Compose final text
    if use_headers:
        report = f"FINDINGS: {' '.join(findings)} IMPRESSION: {' '.join(impression)}"
    else:
        report = " ".join(findings + impression)

    return " ".join(report.split())


# -----------------------------
# RUN
# -----------------------------
df = pd.read_csv(INPUT_CSV)

required = ["Image Index", "Patient ID", "Finding Labels"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")

# Binary label for Effusion (1 if contains Effusion, else 0)
df["label"] = df["Finding Labels"].astype(
    str).str.contains(POS_LABEL, regex=False).astype(int)

# Generate paired synthetic text
df["text"] = df["Finding Labels"].astype(str).apply(generate_synthetic_report)

# Output (keep key columns)
out = df[["Image Index", "Patient ID", "Finding Labels", "text", "label"]].copy()

Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print("Rows:", len(out))
print("Pos (Effusion):", int(out["label"].sum()))
print("Neg:", int((out["label"] == 0).sum()))
print("\nExample positive:\n", out[out["label"] == 1].iloc[0]["text"])
print("\nExample negative:\n", out[out["label"] == 0].iloc[0]["text"])
