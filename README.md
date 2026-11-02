# Ex. No. 3(b) - Data Preprocessing

## Aim
To perform data preprocessing using Python, Pandas and Scikit-learn by handling missing values, encoding categorical variables, splitting data into training and testing sets, and scaling numerical features.

## Software Requirements
- Python 3.x
- Pandas
- NumPy
- Scikit-learn

## Procedure
1. Import the required Python libraries.
2. Create a sample dataset directly using a Pandas DataFrame.
3. Display the first five records and inspect the dataset structure.
4. Separate the independent variables (X) and dependent variable (y).
5. Split the dataset into training and testing sets.
6. Impute missing numerical values using the mean.
7. Impute missing categorical values using the most frequent value.
8. Encode the categorical input variable, Country, using One-Hot Encoding.
9. Encode the target variable, Purchased, using Label Encoding.
10. Standardize numerical features using StandardScaler.
11. Display the preprocessed datasets and their dimensions.

## Dataset
A synthetic dataset containing 15 records and the columns `Country`, `Age`, `Salary`, and `Purchased` is created directly in the program. Missing values are included in the numerical columns to demonstrate imputation.

**No external dataset or CSV file is required.**

## Program

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Create a sample dataset directly in the program.
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

print("First five records:\\n", df.head())
print("\\nDataset information:")
df.info()
print("\\nDataset shape:", df.shape)
print("\\nMissing values:\\n", df.isnull().sum())

# Separate input features and target variable.
X = df.drop(columns=["Purchased"])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Purchased"])

# Split the data before fitting preprocessing steps.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = ["Age", "Salary"]
cat_cols = ["Country"]

# Impute missing numerical values and standardize them.
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Impute missing categorical values and encode them.
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Fit on training data and transform both sets.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\\nPreprocessed training data:\\n", X_train_processed)
print("\\nPreprocessed testing data:\\n", X_test_processed)
print("\\nTraining set shape:", X_train_processed.shape)
print("Testing set shape:", X_test_processed.shape)
print("Training target:", y_train)
print("Testing target:", y_test)
```

## Output

```text
First five records:
   Country   Age   Salary Purchased
0    India  22.0  25000.0        No
1      USA  35.0  65000.0       Yes
2       UK   NaN  55000.0       Yes
3    India  28.0      NaN        No
4  Germany  42.0  80000.0       Yes

Dataset information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 15 entries, 0 to 14
Data columns (total 4 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   Country    15 non-null     object
 1   Age        13 non-null     float64
 2   Salary     13 non-null     float64
 3   Purchased  15 non-null     object
dtypes: float64(2), object(2)
memory usage: 612.0+ bytes

Dataset shape: (15, 4)

Missing values:
Country      0
Age          2
Salary       2
Purchased    0
dtype: int64

Preprocessed training data:
[[-1.1945867  -1.35148438  0.          0.          0.          1.        ]
 [-0.41125116  0.8823741   0.          0.          0.          1.        ]
 [ 0.95958604  0.          1.          0.          0.          0.        ]
 [ 0.         -0.06701575  0.          0.          1.          0.        ]
 [ 0.37208438  0.49144887  0.          0.          0.          1.        ]
 [ 0.56791827  0.65898825  0.          1.          0.          0.        ]
 [ 1.74292158  1.3291458   1.          0.          0.          0.        ]
 [ 0.         -1.46317731  0.          1.          0.          0.        ]
 [-0.80291893 -0.90471268  0.          1.          0.          0.        ]
 [ 1.35125381  1.60837811  1.          0.          0.          0.        ]
 [-0.99875282  0.          0.          1.          0.          0.        ]
 [-1.58625448 -1.18394499  0.          0.          1.          0.        ]]

Preprocessed testing data:
[[ 2.33042324  1.88761042  0.          0.          0.          1.        ]
 [-0.01958339  0.10052363  0.          0.          1.          0.        ]
 [-2.17375613 -1.74240962  0.          1.          0.          0.        ]]

Training set shape: (12, 6)
Testing set shape: (3, 6)
Training target: [0 1 1 1 1 1 1 0 0 1 0 0]
Testing target: [1 1 0]
```

## Result
The synthetic dataset was successfully preprocessed using Pandas and Scikit-learn. Missing values were imputed, categorical variables were encoded, the data was split into training and testing sets, and numerical features were standardized.
