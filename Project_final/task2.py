import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--algorithm",
    "-a",
    help="algorithm",
)

args = parser.parse_args()

algorithm = args.algorithm

dataset = pd.read_csv("Task 2/fuel_emissions.csv")

# take only rows that are not nan on column fuel_cost_12000_miles
dataset = dataset[dataset["fuel_cost_12000_miles"].notna()]


# print(dataset.info())
# print("Unique values per column")
# for column_name in dataset.columns:
#     print("{0} : {1}".format(column_name, len(dataset[column_name].unique())))


keep_columns = [
    column for column in dataset.columns
    if column not in [
        "file", "description", "tax_band", "thc_nox_emissions", "particulates_emissions"
        , "standard_12_months", "standard_6_months", "first_year_12_months", "first_year_6_months", "fuel_cost_12000_miles"]]
X = dataset[keep_columns]
y = dataset['fuel_cost_12000_miles'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True)

# One hot encoding
one_hot_encoding_cols = ['manufacturer', 'model', 'transmission', 'transmission_type', 'fuel_type']
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(X_train[one_hot_encoding_cols]).toarray())
X_train = X_train.join(enc_df).drop(columns=one_hot_encoding_cols)

enc_df = pd.DataFrame(enc.transform(X_test[one_hot_encoding_cols]).toarray())
X_test = X_test.join(enc_df).drop(columns=one_hot_encoding_cols)

print("Imputing...")
# Impute missing values
imputer = SimpleImputer(strategy="mean")
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

print("Normalizing...")
# Normalize feature values using MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


print("Training...")
if algorithm == "svr":
    reg = LinearSVR(C=10, random_state=42, verbose=1, max_iter=10000)
elif algorithm == "tree":
    reg = DecisionTreeRegressor(random_state=42, criterion="poisson")
elif algorithm == "knn":
    reg = KNeighborsRegressor(n_neighbors=5)
elif algorithm == "forest":
    reg = RandomForestRegressor(n_estimators=100, criterion="mse", n_jobs=12)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))
print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
