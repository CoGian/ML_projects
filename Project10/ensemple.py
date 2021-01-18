

# IMPORT NECESSARY LIBRARIES HERE
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# =============================================================================


# Load breastCancer data
# =============================================================================


# ADD COMMAND TO LOAD DATA HERE
breastCancer = load_breast_cancer()

# =============================================================================


# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=44, shuffle=True)

estimators = [
    ('rf', RandomForestClassifier(criterion="entropy", max_depth=6, n_estimators=55, random_state=44)),
    ('svr', make_pipeline(StandardScaler(),
                          SVC(C=10,kernel="rbf", random_state=44)))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(), verbose=1
)

clf.fit(X=x_train, y=y_train)

y_pred = clf.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, average="micro"))
print(recall_score(y_test, y_pred, average="micro"))
print(f1_score(y_test, y_pred, average="micro"))

