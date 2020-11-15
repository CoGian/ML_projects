# =============================================================================
# HOMEWORK 4 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email us: arislaza@csd.auth.gr, ipierros@csd.auth.gr
# =============================================================================

# import the KNeighborsClassifier
# if you want to do the hard task, also import the KNNImputer
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import argparse
from tqdm import tqdm
from collections import defaultdict
import os
random.seed = 42
np.random.seed(666)

parser = argparse.ArgumentParser()
parser.add_argument(
	"--weights",
	"-w",
	help="weights",
)
parser.add_argument(
	"--power",
	"-p",
	help="power parameter of Minkowski metric",
)

args = parser.parse_args()

weights = args.weights
p = int(args.power)

# Import the titanic dataset
# Decide which features you want to use (some of them are useless, ie PassengerId).
#
# Feature 'Sex': because this is categorical instead of numerical, KNN can't deal with it, so drop it
# Note: another solution is to use one-hot-encoding, but it's out of the scope for this exercise.
#
# Feature 'Age': because this column contains missing values, KNN can't deal with it, so drop it
# If you want to do the harder task, don't drop it.
#
# =============================================================================
titanic = pd.read_csv("titanic.csv")
titanic = titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
# one-hot-encoding sex
titanic = pd.concat(
	objs=[titanic, pd.get_dummies(data=titanic['Sex'], prefix='category')], axis=1).drop(columns=['Sex'])
# one-hot-encoding Embarked
titanic = pd.concat(
	objs=[titanic, pd.get_dummies(data=titanic['Embarked'], prefix='embarked')], axis=1).drop(columns=['Embarked'])

na_titanic = titanic.drop(columns=['Age'])

x_cols = [column for column in titanic.columns if column != 'Survived']
X = titanic[x_cols]
y = titanic['Survived']

na_x_cols = [column for column in na_titanic.columns if column != 'Survived']
na_X = na_titanic[na_x_cols]
na_y = na_titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.20, random_state=42, shuffle=True)
na_X_train, na_X_test, na_y_train, na_y_test = train_test_split(
	na_X, na_y, test_size=0.20, random_state=42, shuffle=True)

# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

na_scaler = MinMaxScaler(feature_range=(0, 1))
na_scaler.fit(na_X_train)
na_X_train = na_scaler.transform(na_X_train)
na_X_test = na_scaler.transform(na_X_test)
# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer. 
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================
imputer = KNNImputer(n_neighbors=3)
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Create your KNeighborsClassifier models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================
classifiers_res = defaultdict(dict)
print("Test KNN")
for k in tqdm(range(1, 201)):
	knn_clf = KNeighborsClassifier(n_neighbors=k, weights=weights, p=p, metric='minkowski')
	knn_clf.fit(X_train, y_train)
	y_pred = knn_clf.predict(X_test)
	classifiers_res[k]['accuracy'] = accuracy_score(y_true=y_test, y_pred=y_pred)
	classifiers_res[k]['precision'] = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
	classifiers_res[k]['recall'] = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
	classifiers_res[k]['f1'] = f1_score(y_true=y_test, y_pred=y_pred, average='macro')

res_df = pd.DataFrame.from_dict(classifiers_res).transpose()

na_classifiers_res = defaultdict(dict)
print("Test KNN without filling na values")
for k in tqdm(range(1, 201)):
	knn_clf = KNeighborsClassifier(n_neighbors=k, weights=weights, p=p, metric='minkowski')
	knn_clf.fit(na_X_train, na_y_train)
	na_y_pred = knn_clf.predict(na_X_test)
	na_classifiers_res[k]['accuracy'] = accuracy_score(y_true=na_y_test, y_pred=na_y_pred)
	na_classifiers_res[k]['precision'] = precision_score(y_true=na_y_test, y_pred=na_y_pred, average='macro')
	na_classifiers_res[k]['recall'] = recall_score(y_true=na_y_test, y_pred=na_y_pred, average='macro')
	na_classifiers_res[k]['f1'] = f1_score(y_true=na_y_test, y_pred=na_y_pred, average='macro')

na_res_df = pd.DataFrame.from_dict(na_classifiers_res).transpose()

if not os.path.exists(weights):
	os.makedirs(weights)
res_df.to_csv(weights + '/' + 'impute_' + weights + '_' + 'minkowski_' + str(p) + '.csv')
na_res_df.to_csv(weights + '/' + 'no_impute_' + weights + '_' + 'minkowski_' + str(p) + '.csv')

# Plot the F1 performance results for any combination οf parameter values of your choice.
# If you want to do the hard task, also plot the F1 results with/without imputation (in the same figure)
# =============================================================================
plt.title('k-Nearest Neighbors (Weights = ' + weights + ', Metric = minkowski' + ', p = ' + str(p) + ' )')
plt.plot(res_df['f1'], label='with impute')
plt.plot(na_res_df['f1'], label='without impute')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('F1')
plt.savefig(weights + '/' + weights + '_' + 'minkowski_' + str(p) + '.png')
