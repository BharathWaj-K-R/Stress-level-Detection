import sys
import numpy as np
import pytest
from pathlib import Path
import joblib
import tempfile
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.evaluate_model import evaluate_model, plot_confusion_matrix, plot_feature_importance

def test_evaluate_model():
    # 1. Create a dummy dataset (10 samples, 16384 features for 128x128 image)
    np.random.seed(42)
    # Generate values in range [0, 1] to represent normalized pixels
    X_dummy = np.random.rand(10, 16384).astype(np.float32)
    y_dummy = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
    
    # 2. Train a small Random Forest on dummy data and save to temp file
    clf = RandomForestClassifier(n_estimators=3, random_state=42)
    clf.fit(X_dummy, y_dummy)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        joblib.dump({
            "model": clf, 
            "label_map": {"0": "Low", "1": "Medium", "2": "High"}, 
            "image_size": 128
        }, model_path)
        
        # 3. Call evaluate_model for both raw_pixels and hog modes
        # Use 2 folds since the dataset size is small
        for mode in ["raw_pixels", "hog"]:
            results = evaluate_model(X_dummy, y_dummy, model_path, mode, cv_folds=2)
            
            # Assertions
            assert "accuracy" in results
            assert "precision_per_class" in results
            assert "recall_per_class" in results
            assert "f1_per_class" in results
            assert "confusion_matrix" in results
            
            assert isinstance(results["accuracy"], float)
            assert len(results["precision_per_class"]) == 3
            assert len(results["recall_per_class"]) == 3
            assert len(results["f1_per_class"]) == 3
            
            cm = results["confusion_matrix"]
            assert len(cm) == 3
            assert len(cm[0]) == 3
            
            # Test plotting confusion matrix
            cm_save_path = Path(tmpdir) / f"cm_{mode}.png"
            plot_confusion_matrix(cm, save_path=cm_save_path)
            assert cm_save_path.exists()
            
            # Test plotting feature importance
            fi_save_path = Path(tmpdir) / f"fi_{mode}.png"
            feature_count = results["feature_count"]
            feature_names = [f"feat_{i}" for i in range(feature_count)]
            plot_feature_importance(results["final_model"], feature_names, save_path=fi_save_path)
            assert fi_save_path.exists()
