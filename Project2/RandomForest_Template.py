# =============================================================================
# HOMEWORK 2 - DECISION TREES
# RANDOM FOREST ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'ensemble' package, for calling the Random Forest classifier
# 'model_selection', (instead of the 'cross_validation' package), which will help validate our results.
# =============================================================================


# IMPORT NECESSARY LIBRARIES HERE
from sklearn import datasets, metrics, ensemble, model_selection
import pandas as pd
import os
import matplotlib.pyplot as plt
# =============================================================================


# Load breastCancer data
# =============================================================================


# ADD COMMAND TO LOAD DATA HERE
breastCancer = datasets.load_breast_cancer()

# =============================================================================


# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

# Split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# This proportion can be changed using the 'test_size' or 'train_size' parameter.
# Alsao, passing an (arbitrary) value to the parameter 'random_state' "freezes" the splitting procedure 
# so that each run of the script always produces the same results (highly recommended).
# Apart from the train_test_function, this parameter is present in many routines and should be
# used whenever possible.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=44, shuffle=True)

# RandomForestClassifier() is the core of this script. You can call it from the 'ensemble' class.
# You can customize its functionality in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the Information Gain.
# 'n_estimators': The number of trees in the forest. The larger the better, but it will take longer to compute. Also,
#                 there is a critical number after which there is no significant improvement in the results
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================


# ADD COMMAND TO CREATE RANDOM FOREST CLASSIFIER MODEL HERE
n_estimators = [1] + [i for i in range(5, 205, 5)]

gini_models = [
	ensemble.RandomForestClassifier(criterion='gini', max_depth=4, n_estimators=num, random_state=4)
	for num in n_estimators]

entropy_models = [
	ensemble.RandomForestClassifier(criterion='entropy', max_depth=6, n_estimators=num, random_state=4)
	for num in n_estimators]

# =============================================================================


# Let's train our model.
# =============================================================================


# ADD COMMAND TO TRAIN YOUR MODEL HERE
for model in gini_models:
	model.fit(X=x_train, y=y_train)

for model in entropy_models:
	model.fit(X=x_train, y=y_train)

# =============================================================================


# Ok, now let's predict the output for the test set
# =============================================================================


# ADD COMMAND TO MAKE A PREDICTION HERE
gini_test_predictions = {model.get_params()['n_estimators']: model.predict(X=x_test) for model in gini_models}
entropy_test_predictions = {model.get_params()['n_estimators']: model.predict(X=x_test) for model in entropy_models}

# =============================================================================


# Time to measure scores. We will compare predicted output (from input of second subset, i.e. x_test)
# with the real output (output of second subset, i.e. y_test).
# You can call 'accuracy_score', 'recall_score', 'precision_score', 'f1_score' or any other available metric
# from the 'sklearn.metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# One of the following can be used for this example, but it is recommended that 'macro' is used (for now):
# 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
# 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
#             This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
# =============================================================================


# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
gini_test_scores = pd.DataFrame(n_estimators, columns=['n_estimators'])
gini_test_scores['accuracy'] = [metrics.accuracy_score(y_true=y_test, y_pred=y_pred) for num, y_pred in
																gini_test_predictions.items()]
gini_test_scores['precision'] = [metrics.precision_score(y_true=y_test, y_pred=y_pred, average='macro') for
																 num, y_pred in
																 gini_test_predictions.items()]
gini_test_scores['recall'] = [metrics.recall_score(y_true=y_test, y_pred=y_pred, average='macro') for num, y_pred in
															gini_test_predictions.items()]
gini_test_scores['f1'] = [metrics.f1_score(y_true=y_test, y_pred=y_pred, average='macro') for num, y_pred in
													gini_test_predictions.items()]
print('FORESTS WITH GINI TEST SCORES:')
gini_test_scores = gini_test_scores.set_index('n_estimators')
# fig = gini_test_scores.plot().get_figure()
# fig.savefig('test.pdf')
print(gini_test_scores.sort_values('f1', ascending=False).head(10))
print()
if not os.path.exists('gini_forests'):
	os.makedirs('gini_forests')
gini_test_scores.to_csv('gini_forests/gini_test_scores.csv')

entropy_test_scores = pd.DataFrame(n_estimators, columns=['n_estimators'])
entropy_test_scores['accuracy'] = [metrics.accuracy_score(y_true=y_test, y_pred=y_pred) for num, y_pred in
																	 entropy_test_predictions.items()]
entropy_test_scores['precision'] = [metrics.precision_score(y_true=y_test, y_pred=y_pred, average='macro') for
																		num, y_pred in
																		entropy_test_predictions.items()]
entropy_test_scores['recall'] = [metrics.recall_score(y_true=y_test, y_pred=y_pred, average='macro') for num, y_pred
																 in
																 entropy_test_predictions.items()]
entropy_test_scores['f1'] = [metrics.f1_score(y_true=y_test, y_pred=y_pred, average='macro') for num, y_pred in
														 entropy_test_predictions.items()]
print('FORESTS WITH ENTROPY TEST SCORES:')
entropy_test_scores = entropy_test_scores.set_index('n_estimators')
print(entropy_test_scores.sort_values('f1', ascending=False).head(10))
print()
if not os.path.exists('entropy_forests'):
	os.makedirs('entropy_forests')
entropy_test_scores.to_csv('entropy_forests/entropy_test_scores.csv')

# =============================================================================


# A Random Forest has been trained now, but let's train more models,
# with different number of estimators each, and plot performance in terms of
# the difference metrics. In other words, we need to make 'n'(e.g. 200) models,
# evaluate them on the aforementioned metrics, and plot 4 performance figures
# (one for each metric).
# In essence, the same pipeline as previously will be followed.
# =============================================================================

# After finishing the above plots, try doing the same thing on the train data
# Hint: you can plot on the same figure in order to add a second line.
# Change the line color to distinguish performance metrics on train/test data
# In the end, you should have 4 figures (one for each metric)
# And each figure should have 2 lines (one for train data and one for test data)


# CREATE MODELS AND PLOTS HERE


# =============================================================================
gini_train_predictions = {model.get_params()['n_estimators']: model.predict(X=x_train) for model in gini_models}
entropy_train_predictions = {model.get_params()['n_estimators']: model.predict(X=x_train) for model in entropy_models}

gini_train_scores = pd.DataFrame(n_estimators, columns=['n_estimators'])
gini_train_scores['accuracy'] = [metrics.accuracy_score(y_true=y_train, y_pred=y_pred)
																 for num, y_pred in gini_train_predictions.items()]
gini_train_scores['precision'] = [metrics.precision_score(y_true=y_train, y_pred=y_pred, average='macro')
																	for num, y_pred in gini_train_predictions.items()]
gini_train_scores['recall'] = [metrics.recall_score(y_true=y_train, y_pred=y_pred, average='macro')
															 for num, y_pred in gini_train_predictions.items()]
gini_train_scores['f1'] = [metrics.f1_score(y_true=y_train, y_pred=y_pred, average='macro')
													 for num, y_pred in gini_train_predictions.items()]
print('FORESTS WITH GINI TRAIN SCORES:')
gini_train_scores = gini_train_scores.set_index('n_estimators')
print(gini_train_scores.sort_values('f1', ascending=False).head(10))
print()

entropy_train_scores = pd.DataFrame(n_estimators, columns=['n_estimators'])
entropy_train_scores['accuracy'] = [metrics.accuracy_score(y_true=y_train, y_pred=y_pred)
																		for num, y_pred in entropy_train_predictions.items()]
entropy_train_scores['precision'] = [metrics.precision_score(y_true=y_train, y_pred=y_pred, average='macro')
																		 for num, y_pred in entropy_train_predictions.items()]
entropy_train_scores['recall'] = [metrics.recall_score(y_true=y_train, y_pred=y_pred, average='macro')
																	for num, y_pred in entropy_train_predictions.items()]
entropy_train_scores['f1'] = [metrics.f1_score(y_true=y_train, y_pred=y_pred, average='macro')
															for n_estimators, y_pred in entropy_train_predictions.items()]
print('FORESTS WITH ENTROPY TRAIN SCORES:')
entropy_train_scores = entropy_train_scores.set_index('n_estimators')
print(entropy_train_scores.sort_values('f1', ascending=False).head(10))


# Plotting
fig, axs = plt.subplots(2, 2)
fig.suptitle('Forests with entropy')
axs[0, 0].plot(entropy_train_scores.index, entropy_train_scores['accuracy'])
axs[0, 0].plot(entropy_test_scores.index, entropy_test_scores['accuracy'])
axs[0, 0].set_title('Accuracy')

axs[0, 1].plot(entropy_train_scores.index, entropy_train_scores['precision'])
axs[0, 1].plot(entropy_test_scores.index, entropy_test_scores['precision'])
axs[0, 1].set_title('Precision')

axs[1, 0].plot(entropy_train_scores.index, entropy_train_scores['recall'])
axs[1, 0].plot(entropy_test_scores.index, entropy_test_scores['recall'])
axs[1, 0].set_title('Recall')

axs[1, 1].plot(entropy_train_scores.index, entropy_train_scores['f1'], label='train')
axs[1, 1].plot(entropy_test_scores.index, entropy_test_scores['f1'],  label='test')
axs[1, 1].set_title('F1')

# set labels
plt.setp(axs[-1, :], xlabel='n_estimators')
plt.legend()
fig.savefig('entropy_forests/forest_with_entropy.png')

# Plotting
fig, axs = plt.subplots(2, 2)
fig.suptitle('Forests with gini')
axs[0, 0].plot(gini_train_scores.index, gini_train_scores['accuracy'])
axs[0, 0].plot(gini_test_scores.index, gini_test_scores['accuracy'])
axs[0, 0].set_title('Accuracy')

axs[0, 1].plot(gini_train_scores.index, gini_train_scores['precision'])
axs[0, 1].plot(gini_test_scores.index, gini_test_scores['precision'])
axs[0, 1].set_title('Precision')

axs[1, 0].plot(gini_train_scores.index, gini_train_scores['recall'])
axs[1, 0].plot(gini_test_scores.index, gini_test_scores['recall'])
axs[1, 0].set_title('Recall')

axs[1, 1].plot(gini_train_scores.index, gini_train_scores['f1'], label='train')
axs[1, 1].plot(gini_test_scores.index, gini_test_scores['f1'],  label='test')
axs[1, 1].set_title('F1')

# set labels
plt.setp(axs[-1, :], xlabel='n_estimators')
plt.legend()
fig.savefig('gini_forests/forest_with_gini.png')

