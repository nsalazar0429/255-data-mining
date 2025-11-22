import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from mpl_toolkits.mplot3d import Axes3D


RULE_INPUT = "outputs/rules.csv"
CBLOF_OUTPUT = "outputs/cblof_scores.csv"

# ~~~~~~ Load the rules CSV and quick clean just in case ~~~~~~~~~
rules_df = pd.read_csv(RULE_INPUT)

print("\nHead: \n", rules_df.head())
print("\nInfo: \n", rules_df.info())

rules_df.dropna(how='all', inplace=True)

if rules_df.empty:
    print("The Rules Data Frame is empty!")

# Identifying numeric columns of the data and droping missing values in numeric columns
numeric_columns = rules_df.select_dtypes(include="number").columns
print("\nNumeric Columns: \n", numeric_columns)

rules_df.dropna(subset=numeric_columns, inplace=True)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~ Feature Prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# extracted 2d matrix for sklearn in a clean and scaled way to be more easily processed in Kmeans
num_features_2dMatrix = rules_df[numeric_columns].values

#standardizes each feature to be consistent in the 2d matrix
scaler = StandardScaler()
num_features_scaled = scaler.fit_transform(num_features_2dMatrix)

print("\nMeans: \n", np.mean(num_features_scaled, axis=0))
print("\nStandard Deviation: \n", np.std(num_features_scaled, axis=0))
print("")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~ K-mean Cluster ~~~~~~~~~~~~~~~~~~~~~~~~~

k_range = range(2, 11)
best_k, best_sil = None, -1.0
silhouette_per_k = []

for k in k_range:
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    except TypeError:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_tmp = km.fit_predict(num_features_scaled)
    if len(np.unique(labels_tmp)) > 1:
        sil = silhouette_score(num_features_scaled, labels_tmp)
    else:
        sil = -1.0
    silhouette_per_k.append(sil)
    if sil > best_sil:
        best_sil, best_k = sil, k

cluster_num = best_k
print(f"[Auto-k] Using k={cluster_num} (best silhouette={best_sil:.3f})")


# cluster_num = 6     #allows for repoducability
RANDOM_STATE = 42   #default number used for random states

# any try excepts allows for adaptability between sklean versions
# in this case it is the starting implementation of K-Mean
try:
    kMean_model = KMeans(n_clusters=cluster_num, random_state=RANDOM_STATE, n_init="auto")
except TypeError:
    kMean_model = KMeans(n_clusters=cluster_num, random_state=RANDOM_STATE, n_init=10)


# essentially categorizing cluster groups 
labels = kMean_model.fit_predict(num_features_scaled)
# helps find where the center of each cluster group
centroid_scaled = kMean_model.cluster_centers_
centroid_original = scaler.inverse_transform(centroid_scaled)


# calculates the estimated size of each cluster and ensuring that there are 4 clusters
cluster_size = np.bincount(labels, minlength=cluster_num)


# logs each cluster's information: size, centroid location and if the cluster are reasonable based on
#   size difference betwene groups and pattern
print("Cluster size:\n", cluster_size)
print("Centroids (scaled):\n", pd.DataFrame(centroid_scaled))
if centroid_original is not None:
    print("Centroids (original units):\n", pd.DataFrame(centroid_original))

# ******* Cluster Size Bar Chart *******
plt.figure()
x = np.arange(len(cluster_size))
plt.bar(x, cluster_size)
plt.xlabel("Cluster label")
plt.ylabel("Size (rows)")
plt.title("K-Means Cluster Sizes")
plt.tight_layout()
plt.savefig("outputs/plot_cluster_sizes.png", dpi=150)
plt.show()

# centroid table to save for record
centroids_df = pd.DataFrame(centroid_original, columns=numeric_columns)
centroids_df.index.name = "Cluster"
centroids_df.to_csv("outputs/centroids_original_units.csv", index=False)
# ***************************************

# stores the labels into the Rules DataFrame
rules_df["kMean_label"] = labels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~ Distances, CBLOF core and labels redux ~~~~~~~~~~~~~~~~

# picks the centriods of each row and computes euclidian distance in scaled space from each point to
#   it's point it is assigned to
assigned = centroid_scaled[labels]
dist_to_centroid = np.linalg.norm(num_features_scaled - assigned, axis=1)

# stores the data of each row distance into the Rules DataFrame
rules_df["dist_to_centroid"] = dist_to_centroid

# gets creates an array of clusters sizes
cluster_num = kMean_model.n_clusters
cluster_sizes = np.bincount(labels, minlength=cluster_num)

# cblof acts as a CBLOF proxy that mulitpies each points by the size of each cluster
#   best to think, if a point is far from the centroid and is in a larged cluster that point gets a 
#   higher score then stores the data in the Rules DataFrame
cblof = cluster_sizes[labels] * dist_to_centroid
rules_df["cblof"] = cblof

# ********* distance histogram ********
plt.figure()
plt.hist(rules_df["dist_to_centroid"], bins=50)
plt.xlabel("Distance to Centroid Histogram(scaled)")
plt.ylabel("Count")
plt.title("Distribution of Distances to Centroid")
plt.tight_layout()
plt.savefig("outputs/plot_distance_hist.png", dpi=150)
plt.show()
# ***************************************

# this is a threshold helper that provides 3 ways to decide which points are anomolies
def label_anomalies_cblof(cblof_scores, method="highest_scores", highest_scores=22, percent=95, k=3.0):
    # consolidates all scores of each point for ease of processing
    s = np.asarray(cblof_scores)

    if method == "highest_scores":
        # sorts all points by size largest to smallest and chooses the highest ones 
        #   marking potential anaomolies = true and normal = false

        highest_id = np.argsort(s)[-highest_scores:]
        flags = np.zeros_like(s, dtype=bool)
        flags[highest_id] = True
        threshold = s[highest_id].min() if highest_scores > 0 else np.inf

    elif method == "percent":
        # sets what percent of the data points you want to be normal
        #   in our case 95% is normoal points wile 5% are anomalies

        threshold = np.percentile(s, percent)
        flags = s >= threshold
        
    elif method == "stat":
        # computes the average score/mean and the standard deviation (how spread out the scores are)
        #   anything farther than the average margins are considered anomolies

        mu, sigma = s.mean(), s.std(ddof=0)
        threshold = mu + k * sigma
        flags = s >= threshold

    else:
        raise ValueError("method must be 'highest_scores', 'percentile', or 'stat'")
    return flags, threshold

# selects the top 22 highest scoring points and stores the data into Rules DataFrame
anomaly_flags, cblof_threshold = label_anomalies_cblof(
    rules_df["cblof"].values,
    method="highest_scores",
    highest_scores=22,
    percent=95,
    k=3.0
)
rules_df["is_anomalous"] = anomaly_flags
print(f"Anomalies flagged: {rules_df['is_anomalous'].sum()}, threshold={cblof_threshold:.4f}")

# *** Threshold Histogram after Anomoly detection ***
plt.figure()
plt.hist(rules_df["cblof"], bins=50)
plt.axvline(cblof_threshold, color='r', linestyle="--")
plt.xlabel("CBLOF Score")
plt.ylabel("Count")
plt.title(f"CBLOF Distribution (threshold = {cblof_threshold:.2f})")
plt.tight_layout()
plt.savefig("outputs/plot_cblof_hist.png", dpi=150)
plt.show()

# ********* Anomolies Scattrter Plot **************
pca = PCA(n_components=2, random_state=42)
points_2d = pca.fit_transform(num_features_scaled)
centroids_2d = pca.transform(centroid_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(points_2d[~rules_df["is_anomalous"], 0],
            points_2d[~rules_df["is_anomalous"], 1],
            color='blue', s=5, alpha=0.3, label='Normal')
plt.scatter(points_2d[rules_df["is_anomalous"], 0],
            points_2d[rules_df["is_anomalous"], 1],
            color='red', s=30, alpha=0.9, label='Anomaly')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Clusters with Anomalies Highlighted")
plt.legend()
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/plot_clusters_normals_vs_anomalies.png", dpi=200)
plt.show()
# ****************************************************



rules_df.to_csv("outputs/rules_with_cblof.csv", index=False)
pd.DataFrame({"cluster": np.arange(cluster_num), "size": cluster_sizes}).to_csv(
    "outputs/kMean_cluster_sizes.csv", index=False
)

pca = PCA(n_components=2, random_state=42)
points_2d = pca.fit_transform(num_features_scaled)
centroids_2d = pca.transform(centroid_scaled)

# ********* TEST ********
print("--- Diagnostics ---")
print("Rows after cleaning:", len(rules_df))
print("points_2d shape:", points_2d.shape)
print("Anomalies:", rules_df["is_anomalous"].sum(),
      "Normals:", (~rules_df["is_anomalous"]).sum())
print("Unique clusters:", np.unique(labels))
print("Cluster sizes:", np.bincount(labels))
assert points_2d.shape[0] == len(rules_df), "Mismatch: points_2d vs rules_df rows"

# ********* 2D PCA scatter plot ********
plt.figure()
plt.scatter(points_2d[:, 0], points_2d[:, 1], s=10, alpha=0.6, c=labels)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=120, marker="X", edgecolors="black")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Clusters in PCA Space")
plt.tight_layout()
plt.savefig("outputs/plot_pca_clusters.png", dpi=150)
plt.show()
# ********* 3D PCA scatter plot ********
pca3d = PCA(n_components=3, random_state=42)
points_3d = pca3d.fit_transform(num_features_scaled)
centroids_3d = pca3d.transform(centroid_scaled)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points_3d[~rules_df["is_anomalous"], 0],
           points_3d[~rules_df["is_anomalous"], 1],
           points_3d[~rules_df["is_anomalous"], 2],
           color='blue', s=10, alpha=0.3, label='Normal')

ax.scatter(points_3d[rules_df["is_anomalous"], 0],
           points_3d[rules_df["is_anomalous"], 1],
           points_3d[rules_df["is_anomalous"], 2],
           color='red', s=30, alpha=0.9, label='Anomaly')

ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2],
           color='black', marker='*', s=200, label='Centroid')

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.set_title("3D PCA View of Clusters and Anomalies")
ax.legend(loc='best')

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/plot_clusters_3d.png", dpi=200)
plt.show()
# ***************************************

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~ Building a clean CBLOF score table ~~~~~~~~~~~~~~~~~

# assigning more descriptive columns and and builds a map of cluster label by size per row
id_cols = rules_df.select_dtypes(exclude=np.number).columns.tolist()
cluster_size_map = pd.Series(cluster_sizes, index=np.arange(cluster_num))
per_row_cluster_size = cluster_size_map.loc[labels].values

# assembles a compact readable output with the following key columns
cblof_scores_df = pd.DataFrame(index=rules_df.index)
if id_cols:
    cblof_scores_df[id_cols] = rules_df[id_cols]
cblof_scores_df["cluster_label"] = rules_df["kMean_label"]
cblof_scores_df["cluster_size"] = per_row_cluster_size
cblof_scores_df["distance_to_centroid"] = rules_df["dist_to_centroid"]
cblof_scores_df["cblof_score"] = rules_df["cblof"]
cblof_scores_df["is_anomaly"] = rules_df["is_anomalous"].astype(bool)
os.makedirs(os.path.dirname(CBLOF_OUTPUT), exist_ok=True)

# sanity check to make sure same number of rows
expected_rows = len(rules_df)
actual_rows = len(cblof_scores_df)
if actual_rows != expected_rows:
    raise RuntimeError(f"Row count mismatch: expected {expected_rows}, got {actual_rows}")

# sanity check to make sure anomoly data matches
expected_highest_scores = 22
actual_highest_scores = int(cblof_scores_df["is_anomaly"].sum())
if actual_highest_scores != expected_highest_scores:
    print(f"Warning: expected {expected_highest_scores} anomalies, found {actual_highest_scores}")

# writes newly cleaned and vetted data to CBLOF Scores DataFrame
cblof_scores_df.to_csv(CBLOF_OUTPUT, index=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~ Silhouette score ~~~~~~~~~~~~~~~~~~~~~~~~~~

try:
    sil = silhouette_score(num_features_scaled, labels)
except Exception:
    sil = float("nan")

print("CBLOF completed.")
print(f"Silhouette Score: {sil:.2f}")

# ********* Silhouette Score Histogram ********
try:
    sil_samples = silhouette_samples(num_features_scaled, labels)
    plt.figure()
    plt.hist(sil_samples, bins=50)
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Count")
    plt.title(f"Silhouette Distribution (mean = {np.mean(sil_samples):.2f})")
    plt.tight_layout()
    plt.savefig("outputs/plot_silhouette_hist.png", dpi=150)
    plt.show()
except Exception as e:
    print(f"Could not compute silhouette samples: {e}")
# **********************************************

print(f"Detected {actual_highest_scores} anomalous rules.")
print(f"Saved results to {CBLOF_OUTPUT}")

# ********* Anomolies per cluster ************
anoms_by_cluster = (
    pd.DataFrame({"label": labels, "is_anom": rules_df["is_anomalous"].astype(bool)})
      .groupby("label")["is_anom"].sum()
      .reindex(np.arange(len(cluster_sizes)), fill_value=0)
)

plt.figure()
plt.bar(np.arange(len(cluster_sizes)), anoms_by_cluster)
plt.xlabel("Cluster")
plt.ylabel("Anomaly count")
plt.title("Anomalies per Cluster")
plt.tight_layout()
plt.savefig("outputs/plot_anomalies_per_cluster.png", dpi=150)
plt.show()
# **********************************************

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~