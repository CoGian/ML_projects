# =============================================================================
# MACHINE LEARNING
# EXAMS - FEBRUARY 2021
# PROGRAMMING PROJECT
# Complete the missing code by implementing the necessary commands.
# =============================================================================

# Libraries to use
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
# ADD COMMAND TO LOAD DATA HERE
breastCancer = load_breast_cancer()

# =============================================================================


# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=83, shuffle=True, stratify=y, test_size=0.4)


clf = KNeighborsClassifier(n_neighbors=15, weights="distance", p=2, metric='minkowski')

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)


acc = accuracy_score(y_test, y_pred) * 100
prec = precision_score(y_test, y_pred, average="macro") * 100
rec = recall_score(y_test, y_pred, average="macro") * 100
f1 = f1_score(y_test, y_pred, average="macro") * 100

print("Acc: ", accuracy_score(y_test, y_pred) * 100)
print("Prec: ", precision_score(y_test, y_pred, average="macro") * 100)
print("Rec: ", recall_score(y_test, y_pred, average="macro") * 100)
print("f1: ", f1_score(y_test, y_pred, average="macro") * 100)


labels = ['Acc', 'Precision', 'Recall', 'f1']
test = [acc, prec, rec, f1]

y__train_pred = clf.predict(x_train)

acc = accuracy_score(y_train, y__train_pred) * 100
prec = precision_score(y_train, y__train_pred, average="macro") * 100
rec = recall_score(y_train, y__train_pred, average="macro") * 100
f1 = f1_score(y_train, y__train_pred, average="macro") * 100

train = [acc, prec, rec, f1]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, test, width, label='test')
rects2 = ax.bar(x + width/2, train, width, label='train')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by metric and dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()