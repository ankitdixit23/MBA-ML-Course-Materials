# Import relevant packages
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

import warnings
np.warnings = warnings # Fix for compatibility issue with pyclustering

from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric

import requests
from io import BytesIO
import os

# switch to turn off plots (Switch to True to turn on plots)
dpl = False

# Get the directory where this script is located for saving plots
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the data
dropbox_url = "https://www.dropbox.com/scl/fi/ezlo7pundle63l8zjgwkn/scotchChar.csv?rlkey=3uvix57rxcmm3iwsxjqa88coy&dl=1"
response = requests.get(dropbox_url)
df = pd.read_csv(BytesIO(response.content))
print(df.head())

name_col = "NAME"
X = df.iloc[:, 2:70].copy().astype(int)
names = df[name_col].astype(str).tolist()

# There's a typo. Change "Ardberg" -> "Ardbeg"
names = ["Ardbeg" if n == "Ardberg" else n for n in names]
df[name_col] = names

# Compute Jaccard distances between all observations
D = pairwise_distances(X.values, metric="jaccard")

# Get distances to Ardbeg
ard_idx = df[name_col].tolist().index("Ardbeg")

# Extract the row of distances from Ardbeg to all others
D_ardbeg = D[ard_idx, :]

# Put it in a Series with names for readability
dist_from_ardbeg = pd.Series(D_ardbeg, index=names)

print(dist_from_ardbeg.sort_values().head(10))

# A little function to pull out names of Whiskys in a target cluster
def list_cluster_mates(cluster_labels, target_name, names):
    t = names.index(target_name)
    c = cluster_labels[t]
    mates = [names[i] for i in np.where(cluster_labels == c)[0]]
    return c, mates

# Help summarize clusters
def print_top_features_by_cluster(X, labels, topn=5, title="Top features per cluster"):
    unique_clusters = np.unique(labels)
    print(f"\n=== {title} ===")
    for c in unique_clusters:
        mask = (labels == c)
        means = X.loc[mask].mean().sort_values(ascending=False)
        top = means.head(topn)
        print(f"\nCluster {c} (n={mask.sum()})")
        for feat, m in top.items():
            print(f"  {feat}: {m:.3f}")

# Using pyclustering to do k-means with Jaccard. Should probably just do Euclidean,
# but want to illustrate.

def jaccard_distance(u, v):
    """Jaccard distance for binary 0/1 vectors u, v."""
    u = np.asarray(u, dtype=bool)
    v = np.asarray(v, dtype=bool)
    inter = np.logical_and(u, v).sum()
    union = np.logical_or(u, v).sum()
    if union == 0:
        return 0.0
    return 1.0 - inter / union

_kmeans_cache = {}  # key: k, value: (labels, centers, obj)

def kmeans_pyclustering_jaccard(Xbin, k, n_init=10, max_iter=200, seed=123):
    if k in _kmeans_cache:
        return _kmeans_cache[k]

    data = Xbin.values.tolist()
    metric = distance_metric(type_metric.USER_DEFINED, func=jaccard_distance)

    rng = np.random.default_rng(seed)
    random.seed(seed)

    best_labels = None
    best_centers = None
    best_obj = np.inf

    for _ in range(n_init):
        # k-means++ initialization under Jaccard
        init_centers = kmeans_plusplus_initializer(data, k, metric=metric).initialize()

        algo = kmeans(data, init_centers, metric=metric, itermax=max_iter, tolerance=1e-6)
        algo.process()

        clusters = algo.get_clusters()
        centers = algo.get_centers()

        # Build labels and compute objective
        labels = np.empty(len(data), dtype=int)
        obj = 0.0
        for cid, inds in enumerate(clusters):
            c = centers[cid]
            for i in inds:
                labels[i] = cid
                obj += jaccard_distance(data[i], c)

        if obj < best_obj:
            best_obj = obj
            best_labels = labels.copy()
            best_centers = [c.copy() for c in centers]

    _kmeans_cache[k] = (best_labels, best_centers, best_obj)
    return _kmeans_cache[k]

Ks = range(2, 21)
objs = []

n_init = 100      # many random starts. Takes a while, could reduce or parallelize.
seed   = 9202015  # reproducible
max_iter = 200

n = len(X)

for K in Ks:
    print(f"Running K={K}", flush=True)
    labels, centers, obj = kmeans_pyclustering_jaccard(
        X, k=K, n_init=n_init, max_iter=max_iter, seed=seed
    )
    objs.append(obj)

elbow_df = pd.DataFrame({"K": list(Ks), "objective": objs})
print(elbow_df)

# --- elbow plot (objective vs K) ---
plt.figure(figsize=(7,4))
plt.plot(elbow_df["K"], elbow_df["objective"], marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Sum of Jaccard distances to centers")
plt.title("Elbow plot (k-means with Jaccard distance)")
plt.xticks(list(Ks))
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "elbow_plot_jaccard.png"), dpi=300)
    plt.close()

for k in [5, 10, 20]:
    labels, centers, obj = _kmeans_cache[k]
    c_id, mates = list_cluster_mates(labels, "Ardbeg", names)
    print(f"Ardbeg cluster ID: {c_id}")
    print("Cluster members:")
    print(", ".join(mates))

labels_k10, centers_k10, obj_k10 = _kmeans_cache[10]
print_top_features_by_cluster(X, labels_k10, topn=5, title="Top 5 features per cluster (k=10)")

# Agglomerative hierarchical clustering on Jaccard + dendrogram with Ardbeg highlighted
D_condensed = squareform(D, checks=False)
Z = linkage(D_condensed, method="complete")

plt.figure(figsize=(12, 6))
d = dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=8, link_color_func=lambda k: 'k')
plt.title("Hierarchical Clustering")
plt.ylabel("Jaccard distance")

# Highlight Ardbeg in red/bold
ax = plt.gca()
for tick in ax.get_xmajorticklabels():
    if tick.get_text() == "Ardbeg":
        tick.set_color("red")
        tick.set_fontweight("bold")

plt.tight_layout()
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "dendrogram.png"), dpi=300)
    plt.close()

# Agglomerative clusters for k = 5, 10, 20
for k in [5, 10, 20]:
    agg = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="complete")
    labs = agg.fit_predict(D)
    c_id, mates = list_cluster_mates(labs, "Ardbeg", names)
    print(f"\n[Agglomerative, k={k}] Ardbeg cluster ID: {c_id}")
    print("Cluster members:")
    print(", ".join(mates))

agg20 = AgglomerativeClustering(n_clusters=20, metric="precomputed", linkage="complete")
labs20 = agg20.fit_predict(D)
print_top_features_by_cluster(X, labs20, topn=5, title="Top 5 features per cluster, Hierarchical (k=10)")


K = 20
clusters = fcluster(Z, t=K, criterion="maxclust")

# To find the distance threshold that yields exactly K clusters:
# Sort the distances where merges occur (Z[:, 2] are the linkage distances)
# For K clusters, the cutoff is just below the (n - K)-th largest merge distance.
n_samples = Z.shape[0] + 1
cut_index = n_samples - K
distance_cutoff = Z[cut_index - 1, 2]

print(f"Distance cutoff for {K} clusters: {distance_cutoff:.4f}")

plt.figure(figsize=(12, 6))
d = dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=8, color_threshold=distance_cutoff)
plt.title("Hierarchical Clustering (20 Clusters)")
plt.ylabel("Jaccard distance")

# Highlight Ardbeg in red/bold
ax = plt.gca()
for tick in ax.get_xmajorticklabels():
    if tick.get_text() == "Ardbeg":
        tick.set_color("red")
        tick.set_fontweight("bold")

if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "dendrogram_20_clusters.png"), dpi=300)
    plt.close()

K = 10
clusters = fcluster(Z, t=K, criterion="maxclust")

# To find the distance threshold that yields exactly K clusters:
# Sort the distances where merges occur (Z[:, 2] are the linkage distances)
# For K clusters, the cutoff is just below the (n - K)-th largest merge distance.
n_samples = Z.shape[0] + 1
cut_index = n_samples - K
distance_cutoff = Z[cut_index - 1, 2]

print(f"Distance cutoff for {K} clusters: {distance_cutoff:.4f}")

plt.figure(figsize=(12, 6))
d = dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=8, color_threshold=distance_cutoff)
plt.title("Hierarchical Clustering (10 Clusters)")
plt.ylabel("Jaccard distance")

# Highlight Ardbeg in red/bold
ax = plt.gca()
for tick in ax.get_xmajorticklabels():
    if tick.get_text() == "Ardbeg":
        tick.set_color("red")
        tick.set_fontweight("bold")

if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "dendrogram_10_clusters.png"), dpi=300)
    plt.close()

K = 5
clusters = fcluster(Z, t=K, criterion="maxclust")

# To find the distance threshold that yields exactly K clusters:
# Sort the distances where merges occur (Z[:, 2] are the linkage distances)
# For K clusters, the cutoff is just below the (n - K)-th largest merge distance.
n_samples = Z.shape[0] + 1
cut_index = n_samples - K
distance_cutoff = Z[cut_index - 1, 2]

print(f"Distance cutoff for {K} clusters: {distance_cutoff:.4f}")

plt.figure(figsize=(12, 6))
d = dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=8, color_threshold=distance_cutoff)
plt.title("Hierarchical Clustering (5 Clusters)")
plt.ylabel("Jaccard distance")

# Highlight Ardbeg in red/bold
ax = plt.gca()
for tick in ax.get_xmajorticklabels():
    if tick.get_text() == "Ardbeg":
        tick.set_color("red")
        tick.set_fontweight("bold")

if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "dendrogram_5_clusters.png"), dpi=300)
    plt.close()

