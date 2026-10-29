# Ex. No. 3(b) - Data Preprocessing

## Aim
To perform data preprocessing using Python, Pandas and Scikit-learn by handling missing values, encoding categorical variables, splitting a dataset into training and testing sets, and scaling numerical features.

## Software Requirements
- Python 3.x
- Pandas
- NumPy
- Scikit-learn

## Procedure
1. Import the required Python libraries.
2. Load the dataset using Pandas.
3. Display the first five records and inspect the dataset structure.
4. Separate the independent variables (X) and dependent variable (y).
5. Split the dataset into training and testing sets.
6. Impute missing numerical values using the mean.
7. Impute missing categorical values using the most frequent value.
8. Encode categorical input variables using One-Hot Encoding.
9. Encode the target variable, `Purchased`, using Label Encoding.
10. Standardize numerical features using `StandardScaler`.
11. Display the preprocessed datasets and their dimensions.

## Dataset
Place `Data.csv` in the same folder as the program. The CSV file should contain these columns: `Country`, `Age`, `Salary`, and `Purchased`.

## Program
The complete program is available in [`data_preprocessing.py`](data_preprocessing.py).

```python
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
```

## Result
The dataset was preprocessed successfully using Pandas and Scikit-learn. Missing values were handled, categorical variables were encoded, the dataset was split into training and testing sets, and numerical features were standardized.
