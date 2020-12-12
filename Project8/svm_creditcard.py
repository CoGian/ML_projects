import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tqdm import tqdm

creditcard_df = pd.read_csv('creditcard.csv')

feature_columns = [column for column in creditcard_df.columns if column != "Class" and column != "Time"]
class_column = "Class"
X_train, X_test, y_train, y_test = train_test_split(
	creditcard_df[feature_columns].values,
	creditcard_df[[class_column]].values.ravel(),
	test_size=0.33,
	random_state=42)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifiers = [
	SVC(C=0.1, kernel="poly", gamma=0.2, degree=2, random_state=42, verbose=1),
	SVC(C=10, kernel="poly", gamma=6, degree=5, random_state=42, verbose=1),
	SVC(C=0.1, kernel="rbf", gamma=0.3, random_state=42, verbose=1),
	SVC(C=10, kernel="rbf", gamma=5, random_state=42, verbose=1),
	SVC(C=0.1, kernel="sigmoid", gamma=0.5, random_state=42, verbose=1),
	SVC(C=10, kernel="sigmoid", gamma=2, random_state=42, verbose=1),
	SVC(C=100, kernel="sigmoid", gamma=5, random_state=42, verbose=1)]

results = defaultdict(dict)

for clf in tqdm(classifiers):
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	params = clf.get_params()
	name = "svm_" + str(params['C']) + "_" + params['kernel'] + "_" + str(params['gamma'])
	if params['kernel'] == 'poly':
		name += str(params['degree'])
	
	results[name]['accuracy'] = accuracy_score(y_true=y_test, y_pred=y_pred)
	results[name]['precision'] = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
	results[name]['recall'] = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
	results[name]['f1'] = f1_score(y_true=y_test, y_pred=y_pred, average='macro')

pd.DataFrame.from_dict(results).transpose().to_csv("res.csv")
