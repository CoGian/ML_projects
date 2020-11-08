# =============================================================================
# HOMEWORK 3 - RULE-BASED LEARNING
# CN2 ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# For this project, the only thing that we will need to import is the "Orange" library.
# However, before importing it, you must first install the library into Python.
# Read the instructions on how to do that (it might be a bit trickier than usual!)
# =============================================================================


# IMPORT LIBRARY HERE (trivial but necessary...)
from Orange.data import Table
from Orange.classification.rules import CN2Learner, CN2UnorderedLearner, EntropyEvaluator, LaplaceAccuracyEvaluator
from Orange.evaluation import CrossValidation, CA, Precision, Recall, F1
import argparse
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import os
# =============================================================================


parser = argparse.ArgumentParser()
parser.add_argument(
	"--category",
	"-c",
	help="category",
)
parser.add_argument(
	"--evaluator",
	"-e",
	help="evaluator",
)
args = parser.parse_args()

category = args.category
evaluator = args.evaluator
# Load 'wine' dataset
# =============================================================================


# ADD COMMAND TO LOAD TRAIN AND TEST DATA HERE
wineData = Table.from_file(filename='wine.csv', sheet='wine')
# =============================================================================


# Define the learner that will be trained with the data.
# Try two different learners: an '(Ordered) Learner' and an 'UnorderedLearner'.
# =============================================================================


# ADD COMMAND TO DEFINE LEARNER HERE
learners = {}

for beam_width in range(3, 5):
	for min_covered_examples in range(7, 8):
		for max_rule_length in range(2, 3):
			if category == 'unordered':
				learner = CN2UnorderedLearner()
			else:
				learner = CN2Learner()

			if evaluator == 'laplace':
				learner.rule_finder.quality_evaluator = LaplaceAccuracyEvaluator()
			else:
				learner.rule_finder.quality_evaluator = EntropyEvaluator()
			learner.rule_finder.search_algorithm.beam_width = beam_width
			learner.rule_finder.general_validator.min_covered_examples = min_covered_examples
			learner.rule_finder.general_validator.max_rule_length = max_rule_length
			learners[
				category + '_' +
				evaluator + '_' +
				str(beam_width) + '_' +
				str(min_covered_examples) + '_' +
				str(max_rule_length)] = learner

# We want to test our model now. The CrossValidation() function will do all the
# work in this case, which includes splitting the whole dataset into train and test subsets,
# then train the model, and produce results.
# So, simply initialize the CrossValidation() object from the 'testing' library
# and call it with input arguments 1) the dataset and 2) the learner.
# Note that the 'learner' argument should be in array form, i.e. '[learner]'.
cv = CrossValidation(k=10, random_state=44)
results = {}
for name, learner in tqdm(learners.items()):
	results[name] = cv(data=wineData, learners=[learner])

# As for the required metrics, you can get them using the 'evaluation.scoring' library.
# The 'average' parameter of each metric is used while measuring scores to perform
# a type of averaging on the data. DON'T WORRY MUCH ABOUT THAT JUST YET (AGAIN). USE EITHER
# 'MICRO' OR 'MACRO' (preferably 'macro', at least for final results).
# =============================================================================


# # ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
scores = defaultdict(dict)
for name, res in results.items():
	scores[name]['acc'] = CA(results=res)[0]
	scores[name]['prec'] = Precision(results=res, average='macro')[0]
	scores[name]['rec'] = Recall(results=res, average='macro')[0]
	scores[name]['f1'] = F1(results=res, average='macro')[0]
	classifier = learners[name](data=wineData)
	rules = '\n'.join([str(rule) for rule in classifier.rule_list])
	scores[name]['rules'] = rules

scores_df = pd.DataFrame.from_dict(data=scores).transpose()
scores_df.to_csv(category + '_' + evaluator + '.csv')
