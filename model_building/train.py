
import os
import warnings
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(BASE_DIR, "deployment", "best_model.joblib")
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

# Load prepared data.
X_train = pd.read_csv(os.path.join(ARTIFACT_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(ARTIFACT_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(ARTIFACT_DIR, "y_train.csv")).squeeze("columns")
y_test = pd.read_csv(os.path.join(ARTIFACT_DIR, "y_test.csv")).squeeze("columns")

# Encode categorical variables using mappings learned from training data only.
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
category_mappings = {}

for col in categorical_cols:
    categories = sorted(X_train[col].dropna().unique().tolist())
    mapping = {category: code for code, category in enumerate(categories)}
    category_mappings[col] = mapping

    X_train[col] = X_train[col].map(mapping).fillna(-1).astype(int)
    X_test[col] = X_test[col].map(mapping).fillna(-1).astype(int)

# Fill numerical missing values using training medians only.
numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()
numeric_medians = {}

for col in numeric_cols:
    median_value = X_train[col].median()
    numeric_medians[col] = median_value
    X_train[col] = X_train[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)

models = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }
    ),
    "Bagging": (
        BaggingClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "max_samples": [0.8, 1.0]
        }
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5]
        }
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.5, 1.0]
        }
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }
    ),
    "XGBoost": (
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        ),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }
    )
}

mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
mlflow.set_experiment("Tourism Prediction")

best_overall_model = None
best_overall_name = None
best_overall_score = -1

for model_name, (model, params) in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    with mlflow.start_run(run_name=model_name):

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model", model_name)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("cv_f1", grid.best_score_)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)

        input_example = X_train.head(5).copy()
        signature = infer_signature(
            input_example,
            best_model.predict(input_example)
        )

        # MLflow 3.x: use name= instead of deprecated name=.
        # signature and input_example prevent model-schema warnings.
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=signature,
            input_example=input_example
        )

        print("Best Parameters:", grid.best_params_)
        print("CV F1:", round(grid.best_score_, 4))
        print("Test Accuracy:", round(test_accuracy, 4))
        print("Test F1:", round(test_f1, 4))
        print(classification_report(y_test, y_pred, zero_division=0))

        if test_f1 > best_overall_score:
            best_overall_score = test_f1
            best_overall_model = best_model
            best_overall_name = model_name

model_bundle = {
    "model": best_overall_model,
    "feature_columns": list(X_train.columns),
    "category_mappings": category_mappings,
    "numeric_medians": numeric_medians
}

joblib.dump(model_bundle, MODEL_PATH)

print("\n" + "=" * 60)
print("OVERALL BEST MODEL")
print("=" * 60)
print("Model:", best_overall_name)
print("Best Test F1:", round(best_overall_score, 4))
print("Model saved to:", MODEL_PATH)
