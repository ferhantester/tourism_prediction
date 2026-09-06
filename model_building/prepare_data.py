
# Import os for creating directories and handling file paths.
import os

# Import pandas for loading and processing the dataset.
import pandas as pd

# Import train_test_split for dividing the dataset into training and testing sets.
from sklearn.model_selection import train_test_split

# Define the path of the tourism dataset.
DATA_PATH = "data/tourism.csv"

# Define the folder where the prepared datasets will be stored.
ARTIFACT_DIR = "artifacts"

# Define the target column to be predicted.
TARGET = "ProdTaken"

# Display a message before loading the dataset.
print("Loading tourism dataset...")

# Load the tourism dataset into a DataFrame.
df = pd.read_csv(DATA_PATH)

# Display the original number of rows and columns.
print("Original shape:", df.shape)

# Remove the unnecessary index column if it exists.
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Remove duplicate records from the dataset.
df = df.drop_duplicates()

# Display the dataset shape after cleaning.
print("Shape after cleaning:", df.shape)

# Separate input features by removing the target and CustomerID columns.
X = df.drop(
    columns=[TARGET, "CustomerID"],
    errors="ignore"
)

# Store the target variable separately.
y = df[TARGET]

# Split the data into 80% training and 20% testing sets.
# Stratification keeps the target-class distribution balanced in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create the artifacts directory if it does not already exist.
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Save the training features.
X_train.to_csv(
    f"{ARTIFACT_DIR}/X_train.csv",
    index=False
)

# Save the testing features.
X_test.to_csv(
    f"{ARTIFACT_DIR}/X_test.csv",
    index=False
)

# Save the training target values.
y_train.to_csv(
    f"{ARTIFACT_DIR}/y_train.csv",
    index=False
)

# Save the testing target values.
y_test.to_csv(
    f"{ARTIFACT_DIR}/y_test.csv",
    index=False
)

# Confirm that data preparation has completed successfully.
print("Data preparation completed successfully.")

# Display the shape of the training features.
print("X_train:", X_train.shape)

# Display the shape of the testing features.
print("X_test :", X_test.shape)

# Display the shape of the training target.
print("y_train:", y_train.shape)

# Display the shape of the testing target.
print("y_test :", y_test.shape)
