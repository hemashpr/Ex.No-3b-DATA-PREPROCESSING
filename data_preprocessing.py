import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Create a sample dataset directly in the program (no CSV file required).
data = {
    "Country": [
        "India", "USA", "UK", "India", "Germany",
        "USA", "UK", "India", "Germany", "USA",
        "India", "UK", "Germany", "USA", "India"
    ],
    "Age": [
        22, 35, None, 28, 42,
        31, 25, None, 38, 45,
        29, 33, 40, 27, 36
    ],
    "Salary": [
        25000, 65000, 55000, None, 80000,
        72000, 35000, 30000, None, 90000,
        40000, 58000, 85000, 32000, 68000
    ],
    "Purchased": [
        "No", "Yes", "Yes", "No", "Yes",
        "Yes", "No", "No", "Yes", "Yes",
        "No", "Yes", "Yes", "No", "Yes"
    ]
}

df = pd.DataFrame(data)

print("First five records:\n", df.head())
print("\nDataset information:")
df.info()
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Separate input features and target.
X = df.drop(columns=["Purchased"])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Purchased"])

# Split before fitting preprocessing steps to avoid data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = ["Age", "Salary"]
cat_cols = ["Country"]

# Impute missing numerical values and standardize numerical features.
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Impute missing categories and apply one-hot encoding.
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Fit only on training data; apply the same transformation to test data.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessed training data:\n", X_train_processed)
print("\nPreprocessed testing data:\n", X_test_processed)
print("\nTraining set shape:", X_train_processed.shape)
print("Testing set shape:", X_test_processed.shape)
print("Training target:", y_train)
print("Testing target:", y_test)
