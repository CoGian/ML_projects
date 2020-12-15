import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

htru_2_df = pd.read_csv('HTRU_2.csv', header=None).rename(columns={8:'label'})

X_train, X_test, y_train, y_test = train_test_split(
    htru_2_df[[0,1,2,3,4,5,6,7]].values, 
    htru_2_df[['label']].values.ravel(), 
    test_size=0.33,
    random_state=42)

rfc = RandomForestClassifier(criterion= 'entropy', random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

acc = accuracy_score(y_true=y_test, y_pred=y_pred)
prec = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
rec = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
auc_score = auc(fpr, tpr)

print('Simple model scores:')
print("acc: {:.2f}, prec: {:.2f}, rec: {:.2f}, f1: {:.2f}, auc: {:.2f}".format(acc, prec, rec, f1, auc_score))

print('Simple model most important features(method1):')
feature_importances_method1 = {'feature_'+ str(idx): imp for idx, imp in enumerate(rfc.feature_importances_)}
print(sorted(feature_importances_method1, key=feature_importances_method1.get , reverse=True))

print('Simple model most important features(method2):')
result = permutation_importance(rfc, X_train, y_train, n_repeats=10, random_state=42)
feature_importances_method2 = {'feature_'+ str(idx): imp for idx, imp in enumerate(result['importances_mean'])}
print(sorted(feature_importances_method2, key=feature_importances_method2.get, reverse=True))

pca = PCA(n_components=4)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

rfc = RandomForestClassifier(criterion= 'entropy', random_state=42)
rfc.fit(X_train_pca, y_train)
y_pred = rfc.predict(X_test_pca)

acc = accuracy_score(y_true=y_test, y_pred=y_pred)
prec = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
rec = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
fpr_pca, tpr_pca , thresholds = roc_curve(y_test, y_pred, pos_label=1)
auc_score_pca = auc(fpr_pca, tpr_pca)

print('Model using pca scores:')
print("acc: {:.2f}, prec: {:.2f}, rec: {:.2f}, f1: {:.2f}, auc: {:.2f}".format(acc, prec, rec, f1, auc_score_pca))

X_train_best4 = np.vstack([X_train[:,0], X_train[:,2], X_train[:,3], X_train[:,5]]).T
X_test_best4 = np.vstack([X_test[:,0], X_test[:,2], X_test[:,3], X_test[:,5]]).T

rfc = RandomForestClassifier(criterion= 'entropy', random_state=42)
rfc.fit(X_train_best4, y_train)
y_pred = rfc.predict(X_test_best4)

acc = accuracy_score(y_true=y_test, y_pred=y_pred)
prec = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
rec = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
fpr_best4, tpr_best4 , thresholds = roc_curve(y_test, y_pred, pos_label=1)
auc_score_best4 = auc(fpr_best4, tpr_best4)
print('Model using best 4 features scores:')
print("acc: {:.2f}, prec: {:.2f}, rec: {:.2f}, f1: {:.2f}, auc: {:.2f}".format(acc, prec, rec, f1, auc_score_best4))

plt.axis([0, 1, 0, 1])
plt.plot([fpr[1]], [tpr[1]], marker='o', markersize=3, color="blue", label="Simple model")
plt.plot([fpr_pca[1]], [tpr_pca[1]], marker='o', markersize=3, color="red", label="Model using pca")
plt.plot([fpr_best4[1]], [tpr_best4[1]], marker='o', markersize=3, color="green", label="Model using best 4 features")
plt.legend()
plt.show()