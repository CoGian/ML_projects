from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def bayesian_learning():
	newsgroups_train = datasets.fetch_20newsgroups(
		subset='train', random_state=4, remove=('headers', 'footers', 'quotes'))

	newsgroups_test = datasets.fetch_20newsgroups(
		subset='test', random_state=4, remove=('headers', 'footers', 'quotes'))

	vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

	train_vectors = vectorizer.fit_transform(newsgroups_train.data)
	test_vectors = vectorizer.transform(newsgroups_test.data)

	mnb = MultinomialNB()
	parameters = {'alpha': [.001, .004, .01, .04, .1, .4, 1, 4, 10]}
	clf = GridSearchCV(mnb, parameters, verbose=2, cv=10, n_jobs=-1)
	clf.fit(train_vectors, newsgroups_train.target)

	print("Best estimator: ", clf.best_estimator_)

	pred = clf.predict(test_vectors)
	acc = metrics.accuracy_score(newsgroups_test.target, pred)
	prec = metrics.precision_score(newsgroups_test.target, pred, average='macro')
	rec = metrics.recall_score(newsgroups_test.target, pred, average='macro')
	f1 = metrics.f1_score(newsgroups_test.target, pred, average='macro')

	print("acc: ", acc)
	print("prec: ", prec)
	print("rec: ", rec)
	print("F1: ", f1)

	cf_matrix = confusion_matrix(newsgroups_test.target, pred)
	df_cm = pd.DataFrame(cf_matrix, columns=newsgroups_test.target_names, index=newsgroups_test.target_names)

	plt.figure(figsize=(20, 10))
	sns.set(font_scale=1.4)
	fig_hm = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g').get_figure()
	plt.title(
		"Multinomial NB - Confusion matrix (alpha= {:.3f})[Prec: {:f}, Rec: {:f}, F1: {:f}]".format(
			clf.best_params_['alpha'], prec, rec, f1), fontsize=20)
	fig_hm.savefig("cm_heatmap.png")


if __name__ == '__main__':
	bayesian_learning()
