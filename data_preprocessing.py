import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("Data.csv")

print("First five records:\n", df.head())
print("\nDataset information:")
df.info()
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

X = df.drop(columns=["Purchased"])
y = LabelEncoder().fit_transform(df["Purchased"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = ["Age", "Salary"]
cat_cols = ["Country"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print("\nPreprocessed training data:\n", X_train)
print("\nPreprocessed testing data:\n", X_test)
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Training target:", y_train)
print("Testing target:", y_test)
