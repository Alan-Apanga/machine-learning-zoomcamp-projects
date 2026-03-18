import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# 1. DATA PREPARATION

# Paths
base_path = Path.cwd()
file_path = base_path / "data" / "data.csv"
df = pd.read_csv(file_path)


# replacing column names with lower caps and spaces with underscore in the column name
df.columns = df.columns.str.lower().str.replace(' ', '_')

# make column data lower case and replace spaces with '_'
string_columns = df.dtypes[df.dtypes == 'str']
string_columns = list(string_columns.index)
for col in string_columns:#
    df[col] = df[col].str.lower().str.replace(' ', '_')



# 2. SETTING UP THE VALIDATION FRAMEWORK

# total records
n =len(df)

# data set split
n_valid = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - sum([n_valid, n_test])

print(f'Total Records : {n}')
print(f'Train + Validation + Test Records Total: {n_train + n_valid + n_test}')
print(n_train, n_valid, n_test)

df_train = df.iloc[:n_train]                 # train split
df_val = df.iloc[n_train:n_train + n_valid]  # validation split
df_test = df.iloc[n_train + n_valid:]        # test split

# shuffle the data
np.random.seed(2)

np.arange(n)
index = np.arange(n)
np.random.shuffle(index)

train_split = index[: n_train]               # 60% for training by index
val_split = index[n_train:n_train + n_valid] # 20 % validation by index
test_split = index[n_train + n_valid:]       # 20% test by index

df_train = df.iloc[train_split] # train split, 60%
df_val = df.iloc[val_split]     # validation split, 20%
df_test = df.iloc[test_split]   # test split, 20%

# reset indices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# natural logarithm of the target variable to make it more normally distributed
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

# Remove target variable from the train, validation and test data partitions
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']

# 3. TRAINING A LINEAR REGRESSION MODEL

# Dot product of a matrix
def dot(xi, w):
    n = len(xi)
    res = 0.0

    for j in range(n):
        # res = res + w[j] * xi[j]   # see below formula
        res += w[j] * xi[j]
    return res


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0]) # Bias term vector values
    X = np.column_stack([ones, X])  # adds 1's to the 1st column of X

    # Gram Matrix
    
    # XTX_inv.dot(XTX).round(1) # should give an identity matrix
    
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.pinv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y) # Calculation of the weight formula

    w_0 = w_full[0] # bias term
    w = w_full[1:]  # weights

    return w_0, w


# 4. CAR PRICE BASELINE MODEL

# subset of numeric columns used
base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']



def prepare_Xbase(df):  
    df_num = df[base] # subset of numeric columns used
    X = df_num.fillna(0).values # missing value replace with zeros
    
    return X


X_train = prepare_Xbase(df_train)
w_0, w = train_linear_regression(X_train, y_train) # weights and bias term from the training data
y_pred = w_0 + X_train.dot(w) # prediction results

# 5. EVALUATION

# function to calculate rmse
def rmse(y, y_pred): 
    se = (y_pred - y) ** 2
    mse = se.mean() 

    return np.sqrt(mse)

print(f'RMSE: {rmse(y_train, y_pred)}')


# 6. VALIDATING THE BASE MODEL

# test the trained model weight with validation set
X_val = prepare_Xbase(df_val)
y_pred = w_0 + X_val.dot(w)


# measure the error of predic validation values vs actual validation 
print(f'Validation baseline RMSE: {rmse(y_val, y_pred)}')


# 7.FEATURE ENGINEERING

categorical_variables = [
    'make','engine_fuel_type','transmission_type', 'driven_wheels', 
    'market_category', 'vehicle_size', 'vehicle_style'    
]

# top 5 most popular cars by each categorical variable
categories = {}
for c in categorical_variables:
    categories[c] = list(df[c].value_counts().head().index) 

def prepare_Xfull(df):
    df = df.copy()
    features = base.copy()
    
    # Age of cars
    df['age'] = df.year.max() - df.year
    features.append('age')

    # Categorical variables

    # Number of Doors 
    for v in [2, 3, 4]:
        df[f'num_doors_{v}'] = (df.number_of_doors == v).astype('int')
        features.append(f'num_doors_{v}')
    
    # Car Make and other categorical variables
    for name, values in categories.items():
        for value in values:
            df[f'{name}_{value}'] = (df[name] == value).astype('int')
            features.append(f'{name}_{value}')
    
    df_num = df[features] # subset of numeric columns used
    X = df_num.fillna(0).values # missing value replace with zeros
    
    return X

# retrained modelweights with additional features
X_train = prepare_Xfull(df_train)
w_0, w = train_linear_regression(X_train, y_train)

# test the retrained model weights with validation set
X_val = prepare_Xfull(df_val)
y_pred = w_0 + X_val.dot(w)

# measure the error of predic validation values vs actual validation
print(f'Validation RMSE with additional features: {rmse(y_val, y_pred)}')


# 8. MODEL WITH REGULARIZATION

def train_linear_regression_reg(X, y, r=0.01):
    ones = np.ones(X.shape[0]) # Bias term vector values
    X = np.column_stack([ones, X])  # adds 1's to the 1st column of X

    # Gram Matrix

    # XTX_inv.dot(XTX).round(1) # should give an identity matrix 
    XTX = X.T.dot(X)
    
    # Add a small number to the diagonal
    XTX = XTX + r * np.eye(XTX.shape[0])  # Larger r → stronger penalty, smaller weights

    XTX_inv = np.linalg.pinv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y) # Calculation of the weight formula

    w_0 = w_full[0] # bias term
    w = w_full[1:]  # weights

    return w_0, w

# retrained modelweights with additional features and regullariztion
X_train = prepare_Xfull(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

# test the retrained model weights with validation set
X_val = prepare_Xfull(df_val)
y_pred = w_0 + X_val.dot(w)

# measure the error of predic validation values vs actual validation
print(f'Validation RMSE with additional features and regularization: {rmse(y_val, y_pred)}')


# 9. REGULARIZATION PARAMETER TUNING

best_r = None
best_score = float('inf')

for r in [0.0, 0.00001, 0.0001, 0.001, 0.01, 1, 10]:
    X_train = prepare_Xfull(df_train)
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    
    # test the trained model weight with validation set
    X_val = prepare_Xfull(df_val)
    y_pred = w_0 + X_val.dot(w)
    
    
    # measure the error of predic validation values vs actual validation 
    score = rmse(y_val, y_pred) 
    print(f"r: {r:<10}, w_0: {w_0:.4f}, Validation RMSE: {score:.4f}") # as r gets larger performance deteriorate

    if score < best_score:
        best_score = score
        best_r = r

print(f"\nBest r = {best_r}, Best validation RMSE = {best_score:.4f}")


r  = 0.0001 # best r
X_train = prepare_Xfull(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    
# test the trained model weight with validation set
X_val = prepare_Xfull(df_val)
y_pred = w_0 + X_val.dot(w)
score = rmse(y_val, y_pred) 
print(f"Final model with r: {r}, w_0: {w_0:.4f}, Validation RMSE: {score:.4f}")


# 10. USING THE MODEL
df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)
X_full_train = prepare_Xfull(df_full_train)

y_full_train = np.concatenate([y_train, y_val])

# train the model
r  = 0.0001 # best r
w_0, w = train_linear_regression_reg(X_full_train, y_full_train, r=r)

# test the trained model weight with test set
X_test = prepare_Xfull(df_test)
y_pred = w_0 + X_test.dot(w)
score = rmse(y_test, y_pred) 
print(f"Final model with r: {r}, w_0: {w_0:.4f}, Test RMSE: {score:.4f}")

# predict the price of a car from the test set
car = df_test.iloc[20].to_dict()  # sampled car data from webform in dict usually
df_car = pd.DataFrame([car])
X_car = prepare_Xfull(df_car)

y_pred = w_0 + X_car.dot(w)  # predict with the car price with the weights
y_pred = y_pred[0]
y_pred = np.expm1(y_pred) # reverse log transformation to get the predicted price
print(f"Predicted price for the sampled car: {y_pred:.2f}")

# actual price of the car
actual_price = np.expm1(y_test[20])
print(f"Actual price for the sampled car: {actual_price:.2f}")

# difference between predicted price and actual price
difference = y_pred - actual_price
print(f"Difference for the sampled car: {difference:.2f}")