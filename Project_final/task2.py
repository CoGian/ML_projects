import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ridge_regression
from sklearn.svm import SVR
import numpy as np
import argparse

dataset = pd.read_csv("Task 2/fuel_emissions.csv")

# take only rows that are not nan on column fuel_cost_12000_miles
dataset = dataset[dataset["fuel_cost_12000_miles"].notna()]

# print(dataset.info())
# print("Unique values per column")
# for column_name in dataset.columns:
# 	print("{0} : {1}".format(column_name, len(dataset[column_name].unique())))

keep_columns = [column for column in dataset.columns if column not in ["file", "description", "fuel_cost_12000_miles"]]
X = dataset[keep_columns]
y = dataset['fuel_cost_12000_miles'].values

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.20, random_state=42, shuffle=True)

# One hot encoding
one_hot_encoding_cols = ['manufacturer', 'model', 'tax_band', 'transmission', 'transmission_type', 'fuel_type']
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(X_train[one_hot_encoding_cols]).toarray())
X_train = X_train.join(enc_df).drop(columns=one_hot_encoding_cols)

enc_df = pd.DataFrame(enc.transform(X_test[one_hot_encoding_cols]).toarray())
X_test = X_test.join(enc_df).drop(columns=one_hot_encoding_cols)

print("Imputing...")
# Impute missing values
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

print("Normalizing...")
# Normalize feature values using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("Training...")
clf = SVR(C=1.0, epsilon=0.2, verbose=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(mean_absolute_error(y_test, y_pred))
print(mean_absolute_percentage_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
