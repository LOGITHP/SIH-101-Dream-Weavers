"""
Smart India Hackathon 2026 - Problem Statement SIH26101
Team: Dream Weavers (AD14)
Module: Competency Assessment Model Training and Inference Pipeline

Trains and compares:
1. Random Forest Classifier
2. Multinomial Logistic Regression

Selects the best model based on factual evaluation metrics,
saves artifacts to `models/`, and provides an easy-to-use inference wrapper.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import prepare_train_test_data, DataPreprocessor
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_model_comparison



class CompetencyModelTrainer:
    """
    Trains and compares multiple classification models for official competency assessment.
    """

    def __init__(self, models_dir="models", plots_dir="models/plots"):
        self.models_dir = models_dir
        self.plots_dir = plots_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def train_and_compare(self, data_path="data/official_training_dataset.csv"):
        print("\n=======================================================")
        print("SIH 2026: Training Competency Assessment Models")
        print("Team: Dream Weavers (AD14)")
        print("=======================================================")

        # 1. Prepare data
        preprocessor, X_train, X_test, y_train, y_test, train_df, test_df = prepare_train_test_data(data_path)

        # Save preprocessor artifact
        preprocessor_path = os.path.join(self.models_dir, "preprocessing.pkl")
        preprocessor.save(preprocessor_path)

        # 2. Define Candidate Models
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                class_weight="balanced",
                random_state=42
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                random_state=42
            )
        }

        results = {}
        trained_models = {}

        # 3. Train and Evaluate each candidate
        for name, clf in models.items():
            print(f"\nTraining {name} on {len(X_train)} samples...")
            clf.fit(X_train, y_train)
            trained_models[name] = clf

            eval_res, y_pred = evaluate_model(clf, X_test, y_test, model_name=name)
            results[name] = eval_res

            print(f"[{name}] Results:")
            print(f"  * Accuracy:           {eval_res['accuracy'] * 100:.2f}%")
            print(f"  * Precision (Macro):  {eval_res['precision_macro']:.4f}")
            print(f"  * Recall (Macro):     {eval_res['recall_macro']:.4f}")
            print(f"  * F1 Score (Macro):   {eval_res['f1_macro']:.4f}")
            print(f"  * F1 Score (Weight):  {eval_res['f1_weighted']:.4f}")

            # Plot confusion matrix
            cm_filename = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
            cm_path = os.path.join(self.plots_dir, cm_filename)
            plot_confusion_matrix(
                eval_res["confusion_matrix"],
                eval_res["class_labels"],
                f"Confusion Matrix: {name}",
                cm_path
            )

        # 4. Comparative visualization
        comp_plot_path = os.path.join(self.plots_dir, "model_comparison.png")
        plot_model_comparison(results, comp_plot_path)

        # 5. Model Selection based on real F1-score & Accuracy
        best_model_name = max(results.keys(), key=lambda k: (results[k]["f1_macro"], results[k]["accuracy"]))
        best_model = trained_models[best_model_name]
        print(f"\n>>> Selected Best Model: {best_model_name} <<<")
        print(f"    (Accuracy: {results[best_model_name]['accuracy']*100:.2f}%, F1-Macro: {results[best_model_name]['f1_macro']:.4f})")

        # Save selected model
        model_save_path = os.path.join(self.models_dir, "competency_model.pkl")
        with open(model_save_path, "wb") as f:
            pickle.dump({
                "model_name": best_model_name,
                "model": best_model,
                "classes": best_model.classes_.tolist()
            }, f)
        print(f"Best model saved to {model_save_path}")

        # Save evaluation summary
        eval_summary_path = os.path.join(self.models_dir, "evaluation_results.json")
        summary_payload = {
            "dataset_info": {
                "total_records": len(train_df) + len(test_df),
                "train_records": len(train_df),
                "test_records": len(test_df),
                "features_count": len(preprocessor.feature_names)
            },
            "best_model": best_model_name,
            "models_evaluated": results
        }
        with open(eval_summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"Evaluation report saved to {eval_summary_path}")

        return best_model, preprocessor, summary_payload


class CompetencyPredictor:
    """
    Production-grade Inference Wrapper for Competency Assessment.
    Loads pre-trained model and preprocessing pipeline.
    """

    def __init__(self, model_path="models/competency_model.pkl", preprocessor_path="models/preprocessing.pkl"):
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or preprocessor artifact missing. Run training first!")

        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.model_name = data["model_name"]
            self.model = data["model"]
            self.classes = data["classes"]

        self.preprocessor = DataPreprocessor.load(preprocessor_path)

    def predict(self, official_profile: dict) -> dict:
        """
        Accepts a dictionary profile for an official and returns predicted competency level,
        confidence, and probabilities for all classes.
        """
        df = pd.DataFrame([official_profile])
        X = self.preprocessor.transform(df)

        pred_class = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(self.model.classes_, probs)}
        confidence = float(np.max(probs))

        return {
            "predicted_competency": pred_class,
            "confidence": round(confidence, 4),
            "class_probabilities": prob_dict,
            "model_used": self.model_name
        }


if __name__ == "__main__":
    trainer = CompetencyModelTrainer()
    trainer.train_and_compare()
