# loocv to automatically evaluate the performance of a random forest classifier
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier

# create dataset
X, y = make_classification(n_samples=100, n_features=10, n_informative=8, n_redundant=2, random_state=1)
# create loocv procedure
cv = LeaveOneOut()
# create model
model = RandomForestClassifier(random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f' % (mean(scores)))

tp = 0
fp = 0
tn = 0
fn = 0
for idx, score in enumerate(scores):
	if score == 1. and y[idx] == 1:
		tp += 1
	elif score == 1. and y[idx] == 0:
		tn += 1
	elif score == 0. and y[idx] == 0:
		fp += 1
	elif score == 0. and y[idx] == 1:
		fn += 1

print("tp: {0}, fp: {1}, tn: {2}, fn: {3}".format(tp, fp, tn, fn))
print("Accuracy from tp, fp, tn, fn: {:.3f}".format((tp + tn) / (tp + fp + tn + fn)))
