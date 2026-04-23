"""
Ransomware Detection Using Machine Learning (Random Forest)
Capstone Project — University of North Georgia
Dataset: Ransomware Dataset 2024 (Zenodo DOI: 10.5281/zenodo.13890887)
Dynamic/Behavioral Feature Analysis
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

DATASET_PATH = "Final_Dataset_without_duplicate.csv"

FEATURE_COLS = [
    "registry_read", "registry_write", "registry_delete", "registry_total",
    "network_threats", "network_dns", "network_http", "network_connections",
    "processes_malicious", "processes_suspicious", "processes_monitored", "total_procsses",
    "files_malicious", "files_suspicious", "files_text", "files_unknown",
    "dlls_calls", "apis"
]

LABEL_COL  = "Class"
POS_LABEL  = "Malware"
OUTPUT_DIR = "output"


# ─────────────────────────────────────────────
#  1. LOAD & PREPARE DATASET
# ─────────────────────────────────────────────

def load_data(path):
    if not os.path.exists(path):
        print(f"\n[ERROR] Dataset not found at: {path}")
        print("  Please place 'Final_Dataset_without_duplicate.csv' in the same folder as this script.")
        raise SystemExit(1)

    df = pd.read_csv(path)
    print(f"      Loaded {len(df):,} samples  |  Columns: {list(df.columns)}")

    # Drop rows with missing values in feature or label columns
    df = df[FEATURE_COLS + [LABEL_COL]].dropna()
    print(f"      After cleaning: {len(df):,} samples")

    label_counts = df[LABEL_COL].value_counts()
    print(f"      Class distribution:\n{label_counts.to_string()}")

    return df


# ─────────────────────────────────────────────
#  2. TRAIN MODEL
# ─────────────────────────────────────────────

def train_model(df):
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
#  3. EVALUATE MODEL
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=POS_LABEL)
    recall    = recall_score(y_test, y_pred, pos_label=POS_LABEL)
    f1        = f1_score(y_test, y_pred, pos_label=POS_LABEL)
    report    = classification_report(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred)

    return {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1_score":  f1,
        "report":    report,
        "cm":        cm,
        "y_pred":    y_pred,
        "y_test":    y_test
    }


# ─────────────────────────────────────────────
#  4. GENERATE REPORT
# ─────────────────────────────────────────────

def generate_report(metrics, X_test, y_test, model, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrix plot
    cm = metrics["cm"]
    labels = sorted(y_test.unique())
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Feature importance plot
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
    plt.title("Feature Importance")
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fi_path)
    plt.close()

    # Confusion matrix values
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # HTML report
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Ransomware Detection Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
    h2 {{ color: #16213e; margin-top: 30px; }}
    .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
    .metric-card {{ background: #1a1a2e; color: white; padding: 20px; border-radius: 8px; text-align: center; }}
    .metric-card .value {{ font-size: 2em; font-weight: bold; color: #00d4ff; }}
    .metric-card .label {{ font-size: 0.9em; margin-top: 5px; }}
    pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0; }}
    .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #888; font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>Ransomware Detection — Random Forest Classifier</h1>
  <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
     <strong>Dataset:</strong> Zenodo 2024 Ransomware Dataset &nbsp;|&nbsp;
     <strong>Test Samples:</strong> {len(y_test):,}</p>

  <h2>Performance Metrics</h2>
  <div class="metrics-grid">
    <div class="metric-card"><div class="value">{metrics['accuracy']*100:.2f}%</div><div class="label">Accuracy</div></div>
    <div class="metric-card"><div class="value">{metrics['precision']*100:.2f}%</div><div class="label">Precision</div></div>
    <div class="metric-card"><div class="value">{metrics['recall']*100:.2f}%</div><div class="label">Recall</div></div>
    <div class="metric-card"><div class="value">{metrics['f1_score']*100:.2f}%</div><div class="label">F1 Score</div></div>
  </div>

  <h2>Confusion Matrix</h2>
  <p>
    <strong>True Positives:</strong> {tp:,} &nbsp;|&nbsp;
    <strong>True Negatives:</strong> {tn:,} &nbsp;|&nbsp;
    <strong>False Positives:</strong> {fp:,} &nbsp;|&nbsp;
    <strong>False Negatives:</strong> {fn:,}
  </p>
  <img src="confusion_matrix.png" alt="Confusion Matrix">

  <h2>Feature Importance</h2>
  <img src="feature_importance.png" alt="Feature Importance">

  <h2>Classification Report</h2>
  <pre>{metrics['report']}</pre>

  <div class="footer">
    University of North Georgia — Cybersecurity Capstone Project<br>
    Dataset: Ransomware Detection Dataset 2024 (Zenodo DOI: 10.5281/zenodo.13890887)
  </div>
</body>
</html>"""

    report_path = os.path.join(output_dir, "ransomware_report.html")
    with open(report_path, "w") as f:
        f.write(html)

    return report_path


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Ransomware Detection — Random Forest Classifier")
    print("  University of North Georgia — Capstone Project")
    print("=" * 60)

    print("\n[1/4] Loading dataset...")
    df = load_data(DATASET_PATH)

    print("\n[2/4] Training model...")
    model, X_train, X_test, y_train, y_test = train_model(df)
    print(f"      Training samples : {len(X_train):,}")
    print(f"      Test samples     : {len(X_test):,}")

    print("\n[3/4] Evaluating...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"\n      Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"      Precision: {metrics['precision']*100:.2f}%")
    print(f"      Recall   : {metrics['recall']*100:.2f}%")
    print(f"      F1 Score : {metrics['f1_score']*100:.2f}%")
    print(f"\n{metrics['report']}")

    print("[4/4] Generating HTML report...")
    report_path = generate_report(metrics, X_test, y_test, model, output_dir=OUTPUT_DIR)
    print(f"      Report saved → {report_path}")
    print("\n✔  Done! Open output/ransomware_report.html in your browser.\n")


if __name__ == "__main__":
    main()
