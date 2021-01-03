import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
import numpy as np

pd.set_option('display.max_colwidth', 30)
train_df = pd.read_csv("Task 1/NSL-KDDTrain.csv")
test_df = pd.read_csv("Task 1/NSL-KDDTest.csv")

# one-hot-encoding
one_hot_encoding_cols = ['protocol_type', 'service', 'flag']
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(train_df[one_hot_encoding_cols]).toarray())
train_df = train_df.join(enc_df).drop(columns=one_hot_encoding_cols)

enc_df = pd.DataFrame(enc.transform(test_df[one_hot_encoding_cols]).toarray())
test_df = test_df.join(enc_df).drop(columns=one_hot_encoding_cols)

X_train = train_df.values
x_cols = [column for column in test_df.columns if column != 'target']
X_test = test_df[x_cols].values
Y_test = test_df['target'].values
Y_test = np.array([1 if target == "attack" else 0 for target in Y_test])

print("x_train_shape: ", X_train.shape)
print("x_test_shape: ", X_test.shape)
print("y_test_shape: ", Y_test.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


kmeans = KMeans(n_clusters=2, random_state=42, verbose=0)
kmeans.fit(X_train)

cluster_0_size = len([label for label in kmeans.labels_ if label == 0])
cluster_1_size = len([label for label in kmeans.labels_ if label == 1])

print(cluster_0_size)
print(cluster_1_size)


# sc = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=42)
# sc.fit(X_train)
# cluster_0_size = len([label for label in sc.labels_ if label == 0])
# cluster_1_size = len([label for label in sc.labels_ if label == 1])
#
# print(cluster_0_size)
# print(cluster_1_size)


ac = AgglomerativeClustering(n_clusters=2)
ac.fit(X_train)
cluster_0_size = len([label for label in ac.labels_ if label == 0])
cluster_1_size = len([label for label in ac.labels_ if label == 1])

print(cluster_0_size)
print(cluster_1_size)