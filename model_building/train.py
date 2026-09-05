
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier


ARTIFACT_DIR = "artifacts"
MODEL_PATH = "deployment/best_model.joblib"


# --------------------------------------------------
# Load prepared data
# --------------------------------------------------

X_train = pd.read_csv(f"{ARTIFACT_DIR}/X_train.csv")
X_test = pd.read_csv(f"{ARTIFACT_DIR}/X_test.csv")
y_train = pd.read_csv(f"{ARTIFACT_DIR}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{ARTIFACT_DIR}/y_test.csv").squeeze()

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# --------------------------------------------------
# Handle categorical variables consistently
# --------------------------------------------------

categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

category_mappings = {}

for col in categorical_cols:

    categories = sorted(
        set(X_train[col].dropna().unique())
        | set(X_test[col].dropna().unique())
    )

    mapping = {
        category: code
        for code, category in enumerate(categories)
    }

    category_mappings[col] = mapping

    X_train[col] = (
        X_train[col]
        .map(mapping)
        .fillna(-1)
        .astype(int)
    )

    X_test[col] = (
        X_test[col]
        .map(mapping)
        .fillna(-1)
        .astype(int)
    )


# --------------------------------------------------
# Handle numerical missing values
# --------------------------------------------------

numeric_cols = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

numeric_medians = {}

for col in numeric_cols:

    median_value = X_train[col].median()
    numeric_medians[col] = median_value

    X_train[col] = X_train[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)


# --------------------------------------------------
# Define models
# --------------------------------------------------

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
            eval_metric="logloss"
        ),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }
    )
}


# --------------------------------------------------
# MLflow setup
# --------------------------------------------------

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Tourism Prediction")


# --------------------------------------------------
# Train and evaluate models
# --------------------------------------------------

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

        # Evaluate on test set
        y_pred = best_model.predict(X_test)

        from sklearn.metrics import accuracy_score, f1_score

        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model", model_name)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("cv_f1", grid.best_score_)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)

        mlflow.sklearn.log_model(
            best_model,
            name="model"
        )

        print("Best Parameters:", grid.best_params_)
        print("CV F1:", round(grid.best_score_, 4))
        print("Test Accuracy:", round(test_accuracy, 4))
        print("Test F1:", round(test_f1, 4))

        # Select the best model using test F1,
        # matching the model-selection approach used in the notebook.
        if test_f1 > best_overall_score:

            best_overall_score = test_f1
            best_overall_model = best_model
            best_overall_name = model_name


# --------------------------------------------------
# Save best model and preprocessing information
# --------------------------------------------------

print("\n" + "=" * 60)
print("OVERALL BEST MODEL")
print("=" * 60)

print("Model:", best_overall_name)
print("Best Test F1:", round(best_overall_score, 4))

model_bundle = {
    "model": best_overall_model,
    "feature_columns": list(X_train.columns),
    "category_mappings": category_mappings,
    "numeric_medians": numeric_medians
}

os.makedirs(
    "deployment",
    exist_ok=True
)

joblib.dump(
    model_bundle,
    MODEL_PATH
)

print("\nBest model bundle saved successfully:")
print(MODEL_PATH)
