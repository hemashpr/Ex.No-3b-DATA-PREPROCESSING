# Ex.No-3b-DATA PREPROCESSING
## Aim
To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset, and applying feature scaling. 
## Procedure
1.	Import the required Python libraries. 
2.	Mount Google Drive and load the dataset using Pandas. 
3.	Display the first few records of the dataset. 
4.	Inspect the dataset using df.info() and df.shape. 
5.	Separate the independent variables (X) and dependent variable (Y). 
6.	Convert the independent variables into an array. 
7.	Identify and handle missing values using SimpleImputer with the mean strategy. 
8.	Encode the categorical Country column using LabelEncoder. 
9.	Apply One-Hot Encoding to convert categorical country values into dummy variables. 
10.	Encode the dependent variable Purchased using LabelEncoder. 
11.	Split the dataset into training and testing sets using train_test_split. 
12.	Apply StandardScaler for feature scaling. 
13.	Display the preprocessed training and testing datasets. 
### Program
# Step 1: Import libraries and load dataset
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
df = pd.read_csv('/content/drive/MyDrive/Datasets/Data.csv')
# Display dataset
df.head()
# Step 2: Check dataset information
df.info()
print(df.shape)

# Step 3: Separate independent and dependent variables
x = df[['Country', 'Age', 'Salary']]
y = df[['Purchased']].values
# Convert X into array
x = df[['Country', 'Age', 'Salary']].values
# Step 4: Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)
# Step 5: Encode categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])
print(x)
# Step 6: One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
x_country = onehotencoder.fit_transform( df.Country.values.reshape(-1, 1)
).toarray()
print(x_country)
# Encode dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)
# Step 7: Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(  x, y, test_size=0.2, random_state=0)
print(x_train)
print(x_test)
print(y_train)
# Step 8: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
print(x_train)
print(x_test)

## Conclusion
Thus, the given dataset was successfully preprocessed by handling missing values, encoding categorical variables, splitting the data into training and testing sets, and performing feature scaling.   

