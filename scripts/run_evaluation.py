import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.evaluate_model import evaluate_model, plot_confusion_matrix, plot_feature_importance


DATA_PATH = BASE_DIR / "model" / "preprocessed_data.pkl"
VERSION_ROOT = BASE_DIR / "model" / "upgraded" / "versions"
REPORTS_DIR = BASE_DIR / "reports"
FEATURE_MODES = ["raw_pixels", "hog"]
STRESS_ORDER = ["low", "medium", "high"]


def normalize_label(label):
    return str(label).strip().lower().replace(" stress", "")


def encode_labels(labels):
    normalized = [normalize_label(label) for label in labels]
    return np.asarray([STRESS_ORDER.index(label) for label in normalized], dtype=np.int64)


def sort_versions(versions):
    return sorted(versions, key=lambda item: tuple(int(part) for part in item.lstrip("v").split(".")))


def get_latest_model_path(feature_mode):
    mode_dir = VERSION_ROOT / feature_mode
    if not mode_dir.exists():
        raise FileNotFoundError(f"No versioned models found for feature mode '{feature_mode}' in {mode_dir}")

    versions = [path.name for path in mode_dir.iterdir() if path.is_dir() and path.name.startswith("v")]
    if not versions:
        raise FileNotFoundError(f"No semantic version directories found for feature mode '{feature_mode}'")

    latest_version = sort_versions(versions)[-1]
    model_dir = mode_dir / latest_version
    return latest_version, model_dir / "stress_model.pkl", model_dir / "metadata.json"


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    X, y = joblib.load(DATA_PATH)
    X = np.asarray(X, dtype=np.float32)
    if X.max() > 1.0:
        X = X / 255.0

    return X, encode_labels(y)


def save_classification_report(feature_mode, version, report_text):
    report_path = REPORTS_DIR / f"classification_report_{feature_mode}_{version}.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def save_summary(summary):
    summary_path = REPORTS_DIR / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()

    print(f"Loaded dataset with shape {X.shape} and labels {y.shape}")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "dataset_size": int(len(X)),
        "comparisons": {},
    }

    for feature_mode in FEATURE_MODES:
        version, model_path, metadata_path = get_latest_model_path(feature_mode)
        print(f"\n=== Evaluating {feature_mode} ({version}) ===")
        results = evaluate_model(X, y, model_path, feature_mode, cv_folds=5)

        report_path = save_classification_report(feature_mode, version, results["classification_report"])
        cm_path = REPORTS_DIR / f"confusion_matrix_{feature_mode}_{version}.png"
        fi_path = REPORTS_DIR / f"feature_importance_{feature_mode}_{version}.png"
        plot_confusion_matrix(results["confusion_matrix"], save_path=cm_path)
        plot_feature_importance(results["final_model"], results["feature_names"], save_path=fi_path)

        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        summary["comparisons"][feature_mode] = {
            "version": version,
            "accuracy": results["accuracy"],
            "feature_count": results["feature_count"],
            "report_path": str(report_path),
            "confusion_matrix_path": str(cm_path),
            "feature_importance_path": str(fi_path),
            "metadata": metadata,
        }

        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Classification report saved to: {report_path}")
        print(f"Confusion matrix saved to: {cm_path}")
        print(f"Feature importance saved to: {fi_path}")

    raw_accuracy = summary["comparisons"]["raw_pixels"]["accuracy"]
    hog_accuracy = summary["comparisons"]["hog"]["accuracy"]
    print("\n=== Accuracy Comparison ===")
    print(f"Raw pixels: {raw_accuracy:.2%}")
    print(f"HOG:        {hog_accuracy:.2%}")
    print(f"Difference: {hog_accuracy - raw_accuracy:+.2%}")

    summary_path = save_summary(summary)
    print(f"Evaluation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
