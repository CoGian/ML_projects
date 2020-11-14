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
import pandas as pd
import argparse

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
parser.add_argument(
	"--fill",
	"-f",
	help="fill or drop column with nan values",
	action="store_true",
	default=False
)

args = parser.parse_args()

weights = args.weights
p = int(args.power)
fill = args.fill

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

if not fill:
	titanic = titanic.drop(columns=['Age'])
	
print(titanic.head(n=5))
print(titanic.describe())
x_cols = [column for column in titanic.columns if column not in 'Survived']
X = titanic[x_cols]
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4, shuffle=True)

print("Train set size: ", len(X_train))
print("Test set size: ", len(X_test))

# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer. 
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================
if fill:
	imputer = KNNImputer(n_neighbors=3)
	imputer.fit(X_train)
	X_train = imputer.transform(X_train)
	X_test = imputer.transform(X_test)

# Create your KNeighborsClassifier models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================


# Plot the F1 performance results for any combination οf parameter values of your choice.
# If you want to do the hard task, also plot the F1 results with/without imputation (in the same figure)
# =============================================================================
# plt.title('k-Nearest Neighbors (Weights = '<?>', Metric = '<?>', p = <?>)')
# plt.plot(f1_impute, label='with impute')
# plt.plot(f1_no_impute, label='without impute')
# plt.legend()
# plt.xlabel('Number of neighbors')
# plt.ylabel('F1')
# plt.show()
