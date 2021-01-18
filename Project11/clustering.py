from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
	"--algorithm",
	"-a",
	help="algorithm",
)

args = parser.parse_args()

algorithm = args.algorithm

iris = load_iris()

# =============================================================================


# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
X = iris.data

algorithms = []
for c in range(2, 6):
	if algorithm == "kmeans":
		algorithms.append(KMeans(n_clusters=c, random_state=42, verbose=0))
	elif algorithm == "birch":
		algorithms.append(Birch(n_clusters=c, threshold=0.6))
	elif algorithm == "gaussian":
		algorithms.append(GaussianMixture(n_components=c, random_state=42, covariance_type="full"))
	else:
		print("wrong algorithm")
		exit(0)

for i, _t in enumerate(algorithms):
	algorithms[i].fit(X)

for clf in algorithms:
	if algorithm == "gaussian":
		labels = clf.predict(X)
	else:
		labels = clf.labels_

	print("Silhouette: ", silhouette_score(X, labels))
