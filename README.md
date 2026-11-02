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
A synthetic dataset containing 15 records and the columns `Country`, `Age`, `Salary`, and `Purchased` is created inside the Python program. Missing values are deliberately included in the numerical columns to demonstrate imputation.

**No external dataset or CSV file is required.**

## Program
The complete, self-contained program is available in [`data_preprocessing.py`](data_preprocessing.py).

Run it using:

```bash
python data_preprocessing.py
```

## Output
The program displays:
- The first five records and dataset information.
- Dataset dimensions and missing-value counts.
- The transformed training and testing feature arrays.
- Training and testing dimensions.
- Label-encoded target values.

With the included dataset and a 20% test split, the expected dimensions are:
- Original dataset: 15 rows × 4 columns
- Training set: 12 rows
- Testing set: 3 rows
- Processed feature arrays: 6 columns each

## Result
The synthetic dataset was successfully preprocessed using Pandas and Scikit-learn. Missing values were imputed, categorical variables were encoded, the data was split into training and testing sets, and numerical features were standardized.
