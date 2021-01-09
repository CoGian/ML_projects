import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--algorithm",
    "-a",
    help="algorithm",
)

args = parser.parse_args()

algorithm = args.algorithm

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

if algorithm == "kmeans" or algorithm == "birch" or algorithm == "gausian":
    X_train = train_df.values

x_cols = [column for column in test_df.columns if column != 'target']
X_test = test_df[x_cols].values
Y_test = test_df['target'].values

print("x_train_shape: ", X_train.shape)
print("x_test_shape: ", X_test.shape)
print("y_test_shape: ", Y_test.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if algorithm == "kmeans":
    clf = KMeans(n_clusters=2, random_state=42, verbose=0)
elif algorithm == "birch":
    clf = Birch(n_clusters=2, threshold=0.6)
elif algorithm == "gausian":
    clf = GaussianMixture(n_components=2, random_state=42, covariance_type="spherical")
else:
    print("wrong algorithm")
    exit(0)

clf.fit(X_train)

if algorithm != "gausian":
    cluster_0_size = len([label for label in clf.labels_ if label == 0])
    cluster_1_size = len([label for label in clf.labels_ if label == 1])
else:
    Y_train = clf.predict(X_train)
    cluster_0_size = len([label for label in Y_train if label == 0])
    cluster_1_size = len([label for label in Y_train if label == 1])

print("cluster 0 size:", cluster_0_size)
print("cluster 1 size:", cluster_1_size)

positive_class = 1
negative_class = 0
if cluster_1_size > cluster_0_size:
    positive_class = 0
    negative_class = 1

Y_test = np.array([positive_class if target == "attack" else negative_class for target in Y_test])

Y_pred = clf.predict(X_test)

print(accuracy_score(Y_test, Y_pred))
