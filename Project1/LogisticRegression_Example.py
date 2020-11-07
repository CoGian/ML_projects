# =============================================================================
# HOMEWORK 1 - Supervised learning
# LOGISTIC REGRESSION ALGORITHM EXAMPLE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# STEP 1
# From sklearn, we will import:
# 'linear_model', which includes the LogisticRegression() method
# 'datasets', for our data
# 'metrics' package, for measuring scores (accuracy score, precision_score, recall_score, and f1_score)
# 'model_selection' which will help validate our results
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model, datasets, metrics, model_selection
# =============================================================================
# Additionally, for plotting our results, we need to import:
# 'pyplot' from package 'matplotlib' for our visualization purposes
# 'numpy', which implements a wide variety of operations
import numpy as np
import matplotlib.pyplot as plt




# STEP 2
# Load iris data from 'datasets' class
# =============================================================================
iris = datasets.load_iris()
X = iris.data[:, :2]  # Load just 2 feature for simplicity and visualization purposes....
y = iris.target
# =============================================================================



# STEP 3
# Create Logistic Regression model. All models behave differently, according to
# their own, model-specific parameter values. For example, parameter 'C' in this
# example indicates the regularazation strength. Refer to the documentation of each
# method to view its own parameters.
# =============================================================================
logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
# =============================================================================



# STEP 4
# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# This proportion can be changed using the 'test_size' or 'train_size' parameter.
# Alsao, passing an (arbitrary) value to the parameter 'random_state' "freezes" the splitting procedure 
# so that each run of the script always produces the same results (highly recommended).
# Apart from the train_test_function, this parameter is present in many routines and should be
# used whenever possible.
# =============================================================================
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 0)
# =============================================================================



# STEP 5
# Let's train our model.
# =============================================================================
logreg.fit(x_train, y_train)
# =============================================================================


# STEP 6
# Ok, now let's predict the output for the test input set
# =============================================================================
y_predicted = logreg.predict(x_test)
# =============================================================================



# STEP 7
# Time to measure scores. We will compare predicted output (from input of second subset, i.e. x_test)
# with the real output (output of second subset, i.e. y_test).
# You can call 'accuracy_score', 'recall_score', 'precision_score', 'f1_score' or any other available metric
# from the 'sklearn.metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# One of the following can be used for this example, but it is recommended that 'macro' is used:
# 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
# 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
#             This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
# =============================================================================
print("Accuracy: %2f" % metrics.accuracy_score(y_test, y_predicted))
print("Recall: %2f" % metrics.recall_score(y_test, y_predicted, average="macro"))
print("Precision: %2f" % metrics.precision_score(y_test, y_predicted, average="macro"))
print("F1: %2f" % metrics.f1_score(y_test, y_predicted, average="macro"))
# =============================================================================




# STEP 8
# Plot the results.
# =============================================================================
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])


# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading='auto')

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
# =============================================================================