import sys
from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Add project root to sys.path to allow importing preprocessing package
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from preprocessing.feature_extraction import batch_extract_features, FeatureExtractionConfig, get_feature_names

def evaluate_model(X, y, model_path, feature_mode, cv_folds=5):
    """
    Loads trained model, extracts features according to feature_mode, 
    runs StratifiedKFold cross-validation, and returns evaluation metrics.
    """
    # Load model and extract the underlying estimator
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        estimator = artifact["model"]
    else:
        estimator = artifact

    # Extract features
    include_handwriting_features = True
    if isinstance(artifact, dict):
        include_handwriting_features = artifact.get("include_handwriting_features", True)
    config = FeatureExtractionConfig(mode=feature_mode, include_handwriting_features=include_handwriting_features)
    print(f"Extracting features for mode '{feature_mode}'...")
    X_features = batch_extract_features(X, config)
    print(f"Feature matrix shape: {X_features.shape}")

    # Set up cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_pred_all = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_features, y), 1):
        X_train, X_val = X_features[train_idx], X_features[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone and train model
        fold_clf = clone(estimator)
        fold_clf.fit(X_train, y_train)

        # Predict
        y_pred_all[val_idx] = fold_clf.predict(X_val)

    # Compute metrics
    accuracy = float(accuracy_score(y, y_pred_all))
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred_all, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y, y_pred_all, labels=[0, 1, 2]).tolist()
    report = classification_report(y, y_pred_all, labels=[0, 1, 2], target_names=['Low', 'Medium', 'High'], zero_division=0)

    # Train a final model on the full feature set to return for plotting feature importance
    final_model = clone(estimator)
    final_model.fit(X_features, y)

    return {
        "accuracy": accuracy,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": cm,
        "classification_report": report,
        "final_model": final_model,
        "feature_count": X_features.shape[1],
        "feature_names": get_feature_names(config),
    }

def plot_confusion_matrix(cm, class_names=['Low', 'Medium', 'High'], save_path=None):
    """
    Plots the confusion matrix as a heatmap and optionally saves it.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = np.array(cm).max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i][j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plots the top 20 feature importances for models that support it (e.g. Random Forest).
    """
    if not hasattr(model, "feature_importances_"):
        print("Model does not have feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_n = min(20, len(importances))
    top_indices = indices[:top_n]

    plt.figure(figsize=(10, 6))
    plt.title("Top 20 Feature Importances")
    plt.bar(range(top_n), importances[top_indices], align="center", color="skyblue")
    plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
