from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

from data_cleaning import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, RawDataCleaner,
                           TARGET)

RANDOM_STATE = 42
DATA_PATH = "data_A.csv"


# ============================================================================
# 1. DataLoader
# ============================================================================
class DataLoader:
    def __init__(self, csv_path: str, target: str = TARGET):
        self.csv_path = csv_path
        self.target = target

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)

    def split(self, df: pd.DataFrame, test_size: float = 0.2):
        y = df[self.target].astype(str)
        X = df.drop(columns=[self.target])
        return train_test_split(X, y, test_size=test_size,
                                random_state=RANDOM_STATE, stratify=y)


# ============================================================================
# 2. DataPreprocessor  (PREPROCESSING)
# ============================================================================
class DataPreprocessor:
    """
    Membangun transformer preprocessing pada data yang SUDAH dibersihkan:
      - numerik    : imputasi median + standardisasi
      - kategorikal: imputasi modus + one-hot encoding
    """
    def __init__(self, numeric_features=None, categorical_features=None):
        self.numeric_features = numeric_features or NUMERIC_FEATURES
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES

    def build(self) -> ColumnTransformer:
        numeric_tf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_tf = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        return ColumnTransformer([
            ("num", numeric_tf, self.numeric_features),
            ("cat", categorical_tf, self.categorical_features),
        ])


# ============================================================================
# 3. ModelTrainer  (TRAINING)
# ============================================================================
class ModelTrainer:
    """Melatih pipeline lengkap = cleaner + preprocessor + estimator."""
    def __init__(self, preprocessor: ColumnTransformer, estimator):
        self.pipeline = Pipeline([
            ("cleaner", RawDataCleaner()),
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

    def train(self, X_train, y_train, sample_weight=None) -> Pipeline:
        if sample_weight is not None:
            self.pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
        else:
            self.pipeline.fit(X_train, y_train)
        return self.pipeline


# ============================================================================
# 4. ModelEvaluator  (EVALUATION) - multi-class
# ============================================================================
class ModelEvaluator:
    @staticmethod
    def evaluate(model: Pipeline, X_test, y_test) -> dict:
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        # ROC-AUC one-vs-rest (butuh probabilitas)
        try:
            proba = model.predict_proba(X_test)
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_test, proba, multi_class="ovr", average="macro")
        except Exception:
            metrics["roc_auc_ovr"] = float("nan")
        return metrics

    @staticmethod
    def confusion(model: Pipeline, X_test, y_test, labels) -> np.ndarray:
        return confusion_matrix(y_test, model.predict(X_test), labels=labels)


# ============================================================================
# 5. TrainingPipeline  (ORKESTRATOR + MLflow)
# ============================================================================
@dataclass
class ModelSpec:
    name: str
    estimator: object
    params: dict = field(default_factory=dict)
    use_sample_weight: bool = False


class TrainingPipeline:
    def __init__(self, csv_path: str = DATA_PATH,
                 experiment_name: str = "credit_score_classification"):
        self.loader = DataLoader(csv_path)
        self.preprocessor_builder = DataPreprocessor()
        self.experiment_name = experiment_name
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_name = None
        self.best_score = -np.inf
        self.results = []
        self.classes_ = None

    def candidate_models(self):
        return [
            ModelSpec(
                "logistic_regression",
                LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE),
                {"max_iter": 2000, "class_weight": "balanced"},
            ),
            ModelSpec(
                "random_forest",
                RandomForestClassifier(n_estimators=300, max_depth=18,
                                       class_weight="balanced_subsample",
                                       random_state=RANDOM_STATE, n_jobs=-1),
                {"n_estimators": 300, "max_depth": 18},
            ),
            ModelSpec(
                "xgboost",
                XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.9,
                              eval_metric="mlogloss", random_state=RANDOM_STATE),
                {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05},
                use_sample_weight=True,
            ),
        ]

    def run(self) -> Pipeline:
        df = self.loader.load()
        X_train, X_test, y_train, y_test = self.loader.split(df)

        # encode target (Poor/Standard/Good -> 0/1/2) untuk konsistensi antar model
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)
        self.classes_ = list(self.label_encoder.classes_)
        labels_enc = list(range(len(self.classes_)))

        mlflow.set_experiment(self.experiment_name)

        for spec in self.candidate_models():
            with mlflow.start_run(run_name=spec.name):
                preprocessor = self.preprocessor_builder.build()
                trainer = ModelTrainer(preprocessor, spec.estimator)

                sw = (compute_sample_weight("balanced", y_train_enc)
                      if spec.use_sample_weight else None)
                model = trainer.train(X_train, y_train_enc, sample_weight=sw)

                metrics = ModelEvaluator.evaluate(model, X_test, y_test_enc)
                cm = ModelEvaluator.confusion(model, X_test, y_test_enc, labels_enc)

                mlflow.log_param("model_type", spec.name)
                for k, v in spec.params.items():
                    mlflow.log_param(k, v)
                mlflow.log_metrics(metrics)
                mlflow.log_dict(
                    {"confusion_matrix": cm.tolist(), "labels": self.classes_},
                    "confusion_matrix.json")
                mlflow.sklearn.log_model(model, name="model",
                                         serialization_format="pickle")

                self.results.append({"model": spec.name, **metrics})
                print(f"[{spec.name:>20}]  "
                      + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

                if metrics["f1_macro"] > self.best_score:
                    self.best_score = metrics["f1_macro"]
                    self.best_model = model
                    self.best_name = spec.name

        self._save_artifacts()
        return self.best_model

    def _save_artifacts(self):
        # bungkus model + label encoder agar inference mengembalikan nama kelas
        bundle = {"model": self.best_model,
                  "label_encoder": self.label_encoder,
                  "classes": self.classes_}
        with open("model.pkl", "wb") as f:
            pickle.dump(bundle, f)

        metadata = {
            "best_model": self.best_name,
            "best_f1_macro": round(self.best_score, 4),
            "target": TARGET,
            "classes": self.classes_,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        }
        with open("model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        pd.DataFrame(self.results).to_csv("model_comparison.csv", index=False)

        print("\n" + "=" * 64)
        print(f"MODEL TERBAIK : {self.best_name}  (F1-macro={self.best_score:.3f})")
        print(f"Kelas         : {self.classes_}")
        print("Tersimpan     : model.pkl, model_metadata.json, model_comparison.csv")
        print("=" * 64)


if __name__ == "__main__":
    TrainingPipeline().run()
