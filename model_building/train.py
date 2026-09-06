
# Import os for directory and file path operations.
import os

# Import joblib for saving the trained model bundle.
import joblib

# Import MLflow for experiment tracking.
import mlflow

# Import MLflow's scikit-learn integration for logging models.
import mlflow.sklearn

# Import pandas for loading and processing datasets.
import pandas as pd

# Import GridSearchCV for hyperparameter tuning.
from sklearn.model_selection import GridSearchCV

# Import the Decision Tree classifier.
from sklearn.tree import DecisionTreeClassifier

# Import ensemble classification algorithms.
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

# Import the XGBoost classifier.
from xgboost import XGBClassifier


# Define the folder containing the prepared datasets.
ARTIFACT_DIR = "artifacts"

# Define the location where the best model bundle will be saved.
MODEL_PATH = "deployment/best_model.joblib"



# Load prepared data


# Load the training features.
X_train = pd.read_csv(f"{ARTIFACT_DIR}/X_train.csv")

# Load the testing features.
X_test = pd.read_csv(f"{ARTIFACT_DIR}/X_test.csv")

# Load the training target and convert it to a Series.
y_train = pd.read_csv(f"{ARTIFACT_DIR}/y_train.csv").squeeze()

# Load the testing target and convert it to a Series.
y_test = pd.read_csv(f"{ARTIFACT_DIR}/y_test.csv").squeeze()

# Display the training and testing data shapes.
print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)



# Handle categorical variables consistently


# Identify all categorical columns in the training data.
categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

# Create a dictionary to store category-to-number mappings.
category_mappings = {}

# Process each categorical column.
for col in categorical_cols:

    # Collect unique categories from both training and testing data.
    categories = sorted(
        set(X_train[col].dropna().unique())
        | set(X_test[col].dropna().unique())
    )

    # Assign an integer code to each category.
    mapping = {
        category: code
        for code, category in enumerate(categories)
    }

    # Store the mapping for later use during prediction.
    category_mappings[col] = mapping

    # Convert training categories to integer values.
    X_train[col] = (
        X_train[col]
        .map(mapping)
        .fillna(-1)
        .astype(int)
    )

    # Apply the same mapping to testing categories.
    X_test[col] = (
        X_test[col]
        .map(mapping)
        .fillna(-1)
        .astype(int)
    )


# Handle numerical missing values


# Identify numerical columns in the training data.
numeric_cols = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

# Create a dictionary to store training-data medians.
numeric_medians = {}

# Process each numerical column.
for col in numeric_cols:

    # Calculate the median using the training data.
    median_value = X_train[col].median()

    # Store the median for use during prediction.
    numeric_medians[col] = median_value

    # Fill missing values in the training data.
    X_train[col] = X_train[col].fillna(median_value)

    # Fill missing values in the testing data using the training median.
    X_test[col] = X_test[col].fillna(median_value)



# Define models


# Define the machine learning models and their hyperparameter grids.
models = {

    # Decision Tree model and parameters to tune.
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }
    ),

    # Bagging model and parameters to tune.
    "Bagging": (
        BaggingClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "max_samples": [0.8, 1.0]
        }
    ),

    # Random Forest model and parameters to tune.
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5]
        }
    ),

    # AdaBoost model and parameters to tune.
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.5, 1.0]
        }
    ),

    # Gradient Boosting model and parameters to tune.
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }
    ),

    # XGBoost model and parameters to tune.
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



# MLflow setup


# Configure MLflow to store experiment information locally.
mlflow.set_tracking_uri("file:./mlruns")

# Create or select the Tourism Prediction MLflow experiment.
mlflow.set_experiment("Tourism Prediction")


# Train and evaluate models


# Initialize the variable for storing the best model.
best_overall_model = None

# Initialize the variable for storing the best model name.
best_overall_name = None

# Start with a very low F1 score for comparison.
best_overall_score = -1


# Train and tune each defined model.
for model_name, (model, params) in models.items():

    # Display a heading for the current model.
    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    # Configure GridSearchCV with 5-fold cross-validation and F1 scoring.
    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    # Start an MLflow run for the current model.
    with mlflow.start_run(run_name=model_name):

        # Train the model and search for the best hyperparameters.
        grid.fit(X_train, y_train)

        # Retrieve the best model found by GridSearchCV.
        best_model = grid.best_estimator_

        # Generate predictions on the testing data.
        y_pred = best_model.predict(X_test)

        # Import evaluation metrics for model performance measurement.
        from sklearn.metrics import accuracy_score, f1_score

        # Calculate test accuracy.
        test_accuracy = accuracy_score(y_test, y_pred)

        # Calculate test F1 score.
        test_f1 = f1_score(y_test, y_pred)

        # Log the model name in MLflow.
        mlflow.log_param("model", model_name)

        # Log the best hyperparameters.
        mlflow.log_params(grid.best_params_)

        # Log the best cross-validation F1 score.
        mlflow.log_metric("cv_f1", grid.best_score_)

        # Log the test accuracy.
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Log the test F1 score.
        mlflow.log_metric("test_f1", test_f1)

        # Save the trained model in the MLflow run.
        mlflow.sklearn.log_model(
            best_model,
            name="model"
        )

        # Display the best hyperparameters and evaluation results.
        print("Best Parameters:", grid.best_params_)
        print("CV F1:", round(grid.best_score_, 4))
        print("Test Accuracy:", round(test_accuracy, 4))
        print("Test F1:", round(test_f1, 4))

        # Select the model with the highest test F1 score.
        if test_f1 > best_overall_score:

            # Store the highest F1 score.
            best_overall_score = test_f1

            # Store the corresponding best model.
            best_overall_model = best_model

            # Store the name of the best model.
            best_overall_name = model_name



# Save best model and preprocessing information


# Display the overall best model heading.
print("\n" + "=" * 60)
print("OVERALL BEST MODEL")
print("=" * 60)

# Display the name of the best-performing model.
print("Model:", best_overall_name)

# Display the best test F1 score.
print("Best Test F1:", round(best_overall_score, 4))

# Bundle the trained model with preprocessing information.
model_bundle = {
    "model": best_overall_model,
    "feature_columns": list(X_train.columns),
    "category_mappings": category_mappings,
    "numeric_medians": numeric_medians
}

# Create the deployment directory if it does not exist.
os.makedirs(
    "deployment",
    exist_ok=True
)

# Save the model bundle for deployment.
joblib.dump(
    model_bundle,
    MODEL_PATH
)

# Confirm that the best model has been saved.
print("\nBest model bundle saved successfully:")

# Display the saved model path.
print(MODEL_PATH)
