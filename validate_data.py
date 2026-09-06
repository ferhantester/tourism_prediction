
import pandas as pd

# Path of the dataset.
DATA_PATH = "data/tourism.csv"

# List of columns expected in the dataset.
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "OwnCar", "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome", "PitchSatisfactionScore", "ProductPitched",
    "NumberOfFollowups", "DurationOfPitch"
]

# Load the dataset.
df = pd.read_csv(DATA_PATH)

# Display dataset details.
print("Dataset loaded successfully.")
print("Shape:", df.shape)

# Check for missing expected columns.
missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]

# Display validation result.
if missing_columns:
    print("Validation FAILED")
    print("Missing columns:", missing_columns)
else:
    print("Validation PASSED")
    print("All expected columns are present.")

# Display dataset information, missing values, and first 5 records.
print("\nDataset Summary:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Records:")
print(df.head())
