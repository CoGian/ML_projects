# =============================================================================
# HOMEWORK 2 - DECISION TREES
# DECISION TREE ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# =============================================================================
# !!! NOTE !!!
# The below import is for using Graphviz!!! Make sure you install it in your
# computer, after downloading it from here:
# https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# After installation, change the 'C:/Program Files (x86)/Graphviz2.38/bin/' 
# from below to the directory that you installed GraphViz (might be the same though).
# =============================================================================
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'tree' package, for creating the DecisionTreeClassifier and using graphviz
# 'model_selection' package, which will help test our model.
# =============================================================================


# IMPORT NECESSARY LIBRARIES HERE
from sklearn import datasets, metrics, tree, model_selection
import pandas as pd
# =============================================================================


# The 'graphviz' library is necessary to display the decision tree.
# =============================================================================
# !!! NOTE !!!
# You must install the package into python as well.
# To do that, run the following command into the Python console.
# !pip install graphviz
# or
# !pip --install graphviz
# or
# pip install graphviz
# or something like that. Google it.
# =============================================================================
import graphviz

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

# DecisionTreeClassifier() is the core of this script. You can customize its functionality
# in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the information gain.
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================


# ADD COMMAND TO CREATE DECISION TREE CLASSIFIER MODEL HERE
max_depth = range(2, 8)
gini_models = [tree.DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=4) for depth in max_depth]
entropy_models = [tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=4) for depth in
									max_depth]
# =============================================================================


# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=44, shuffle=True)

# Let's train our model.
# =============================================================================


# ADD COMMAND TO TRAIN YOUR MODEL HERE
for model in gini_models:
	model.fit(X=x_train, y=y_train)

for model in entropy_models:
	model.fit(X=x_train, y=y_train)

# =============================================================================


# Ok, now let's predict the output for the test input set
# =============================================================================


# ADD COMMAND TO MAKE A PREDICTION HERE

gini_test_predictions = {model.get_params()['max_depth']: model.predict(X=x_test) for model in gini_models}
entropy_test_predictions = {model.get_params()['max_depth']: model.predict(X=x_test) for model in entropy_models}

# =============================================================================


# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'recall_score()', 'precision_score()', 'accuracy_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# =============================================================================


# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
gini_test_scores = pd.DataFrame(max_depth, columns=['max_depth'])
gini_test_scores['accuracy'] = [metrics.accuracy_score(y_true=y_test, y_pred=y_pred) for depth, y_pred in
																gini_test_predictions.items()]
gini_test_scores['precision'] = [metrics.precision_score(y_true=y_test, y_pred=y_pred, average='macro') for
																 depth, y_pred in
																 gini_test_predictions.items()]
gini_test_scores['recall'] = [metrics.recall_score(y_true=y_test, y_pred=y_pred, average='macro') for depth, y_pred in
															gini_test_predictions.items()]
gini_test_scores['f1'] = [metrics.f1_score(y_true=y_test, y_pred=y_pred, average='macro') for depth, y_pred in
													gini_test_predictions.items()]
print('TREES WITH GINI TEST SCORES:')
gini_test_scores = gini_test_scores.set_index('max_depth')
# fig = gini_test_scores.plot().get_figure()
# fig.savefig('test.pdf')
print(gini_test_scores.head(10))
print()
if not os.path.exists('gini_trees'):
	os.makedirs('gini_trees')
gini_test_scores.to_csv('gini_trees/gini_test_scores.csv')

entropy_test_scores = pd.DataFrame(max_depth, columns=['max_depth'])
entropy_test_scores['accuracy'] = [metrics.accuracy_score(y_true=y_test, y_pred=y_pred) for depth, y_pred in
																	 entropy_test_predictions.items()]
entropy_test_scores['precision'] = [metrics.precision_score(y_true=y_test, y_pred=y_pred, average='macro') for
																		depth, y_pred in
																		entropy_test_predictions.items()]
entropy_test_scores['recall'] = [metrics.recall_score(y_true=y_test, y_pred=y_pred, average='macro') for depth, y_pred
																 in
																 entropy_test_predictions.items()]
entropy_test_scores['f1'] = [metrics.f1_score(y_true=y_test, y_pred=y_pred, average='macro') for depth, y_pred in
														 entropy_test_predictions.items()]
print('TREES WITH ENTROPY TEST SCORES:')
entropy_test_scores = entropy_test_scores.set_index('max_depth')
print(entropy_test_scores.head(10))
print()
if not os.path.exists('entropy_trees'):
	os.makedirs('entropy_trees')
entropy_test_scores.to_csv('entropy_trees/entropy_test_scores.csv')
# =============================================================================


# We always predict on the test dataset, which hasn't been used anywhere.
# Try predicting using the train dataset this time and print the metrics 
# to see how much you have overfitted the model
# Hint: try increasing the max_depth parameter of the model

gini_train_predictions = {model.get_params()['max_depth']: model.predict(X=x_train) for model in gini_models}
entropy_train_predictions = {model.get_params()['max_depth']: model.predict(X=x_train) for model in entropy_models}

gini_train_scores = pd.DataFrame(max_depth, columns=['max_depth'])
gini_train_scores['accuracy'] = [metrics.accuracy_score(y_true=y_train, y_pred=y_pred) for depth, y_pred in
																 gini_train_predictions.items()]
gini_train_scores['precision'] = [metrics.precision_score(y_true=y_train, y_pred=y_pred, average='macro') for
																	depth, y_pred in
																	gini_train_predictions.items()]
gini_train_scores['recall'] = [metrics.recall_score(y_true=y_train, y_pred=y_pred, average='macro') for depth, y_pred in
															 gini_train_predictions.items()]
gini_train_scores['f1'] = [metrics.f1_score(y_true=y_train, y_pred=y_pred, average='macro') for depth, y_pred in
													 gini_train_predictions.items()]
print('TREES WITH GINI TRAIN SCORES:')
gini_train_scores = gini_train_scores.set_index('max_depth')
# fig = gini_train_scores.plot().get_figure()
# fig.savefig('test.pdf')
print(gini_train_scores.head(10))
print()

entropy_train_scores = pd.DataFrame(max_depth, columns=['max_depth'])
entropy_train_scores['accuracy'] = [metrics.accuracy_score(y_true=y_train, y_pred=y_pred) for depth, y_pred in
																		entropy_train_predictions.items()]
entropy_train_scores['precision'] = [metrics.precision_score(y_true=y_train, y_pred=y_pred, average='macro') for
																		 depth, y_pred in
																		 entropy_train_predictions.items()]
entropy_train_scores['recall'] = [metrics.recall_score(y_true=y_train, y_pred=y_pred, average='macro') for depth, y_pred
																	in
																	entropy_train_predictions.items()]
entropy_train_scores['f1'] = [metrics.f1_score(y_true=y_train, y_pred=y_pred, average='macro') for depth, y_pred in
															entropy_train_predictions.items()]
print('TREES WITH ENTROPY TRAIN SCORES:')
entropy_train_scores = entropy_train_scores.set_index('max_depth')
print(entropy_train_scores.head(10))

# =============================================================================


# By using the 'export_graphviz' function from the 'tree' package we can visualize the trained model.
# There is a variety of parameters to configure, which can lead to a quite visually pleasant result.
# Make sure that you set the following parameters within the function:
# feature_names = breastCancer.feature_names[:numberOfFeatures]
# class_names = breastCancer.target_names
# =============================================================================


# ADD COMMAND TO EXPORT TRAINED MODEL HERE

for model in gini_models:
	dot_data = tree.export_graphviz(
		model,
		feature_names=breastCancer.feature_names[:numberOfFeatures],
		class_names=breastCancer.target_names,
		filled=True,
		rounded=True,
	)
	# =============================================================================
	
	# The below command will export the graph into a PDF file located within the same folder as this script. If you want
	# to view it from the Python IDE, type 'graph' (without quotes) on the python console after the script has been
	# executed.
	graph = graphviz.Source(dot_data)
	graph.render("gini_trees/breastCancerTreePlot_gini" + str(model.get_params()['max_depth']))

for model in entropy_models:
	dot_data = tree.export_graphviz(
		model,
		feature_names=breastCancer.feature_names[:numberOfFeatures],
		class_names=breastCancer.target_names,
		filled=True,
		rounded=True,
	)
	# =============================================================================
	
	# The below command will export the graph into a PDF file located within the same folder as this script. If you want
	# to view it from the Python IDE, type 'graph' (without quotes) on the python console after the script has been
	# executed.
	graph = graphviz.Source(dot_data)
	graph.render("entropy_trees/breastCancerTreePlot_entropy" + str(model.get_params()['max_depth']))
