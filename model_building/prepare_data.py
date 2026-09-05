
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/tourism.csv"
ARTIFACT_DIR = "artifacts"

TARGET = "ProdTaken"

print("Loading tourism dataset...")

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# Remove unnecessary column if present
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Remove duplicate records
df = df.drop_duplicates()

print("Shape after cleaning:", df.shape)

# Separate features and target
X = df.drop(
    columns=[TARGET, "CustomerID"],
    errors="ignore"
)

y = df[TARGET]

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create artifact directory
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Save datasets
X_train.to_csv(
    f"{ARTIFACT_DIR}/X_train.csv",
    index=False
)

X_test.to_csv(
    f"{ARTIFACT_DIR}/X_test.csv",
    index=False
)

y_train.to_csv(
    f"{ARTIFACT_DIR}/y_train.csv",
    index=False
)

y_test.to_csv(
    f"{ARTIFACT_DIR}/y_test.csv",
    index=False
)

print("Data preparation completed successfully.")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
