import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RULE_INPUT = "outputs/vectorized.csv"
CBLOF_OUTPUT = "outputs/cblof_scores.csv"

# ~~~~~~ Load the rules CSV and quick clean just in case ~~~~~~~~~
rules_df = pd.read_csv(RULE_INPUT)
rules_df.dropna(how='all', inplace=True)

if rules_df.empty:
    print("The Rules Data Frame is empty!")

numeric_columns = rules_df.select_dtypes(include="number").columns
rules_df.dropna(subset=numeric_columns, inplace=True)

# ~~~~~~~~~~~~~~~~~~~~~ Feature Prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
num_features_2dMatrix = rules_df[numeric_columns].values

scaler = StandardScaler()
num_features_scaled = scaler.fit_transform(num_features_2dMatrix)

# ~~~~~~~~~~~~~~~~~~~~~~~~ K-mean Cluster ~~~~~~~~~~~~~~~~~~~~~~~~~
k_range = range(2, 11)
silhouette_per_k = []
best_k= None 
best_sil = -1.0

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

# ******* refine the silhouette-based *******
plt.figure()
plt.plot(k_range, silhouette_per_k, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("outputs/CBLOF_silhouette_vs_k.png", dpi=150)
plt.show()
# ********************************************

cluster_num = best_k
print(f"[Auto-k] Using k={cluster_num} (best silhouette={best_sil:.3f})")

RANDOM_STATE = 42   #default number used for random states

try:
    kMean_model = KMeans(n_clusters=cluster_num, random_state=RANDOM_STATE, n_init="auto")
except TypeError:
    kMean_model = KMeans(n_clusters=cluster_num, random_state=RANDOM_STATE, n_init=10)

labels = kMean_model.fit_predict(num_features_scaled)

centroid_scaled = kMean_model.cluster_centers_
centroid_original = scaler.inverse_transform(centroid_scaled)

cluster_size = np.bincount(labels, minlength=cluster_num)

# ******* Cluster Size Bar Chart *******
plt.figure()
x = np.arange(len(cluster_size))
plt.bar(x, cluster_size)
plt.xlabel("Cluster label")
plt.ylabel("Size (rows)")
plt.title("K-Means Cluster Sizes")
plt.tight_layout()
plt.savefig("outputs/CBLOF_cluster_sizes.png", dpi=150)
plt.show()
# ***************************************

centroids_df = pd.DataFrame(centroid_original, columns=numeric_columns)
centroids_df.index.name = "Cluster"
centroids_df.to_csv("outputs/centroids_original_units.csv", index=False)

rules_df["kMean_label"] = labels

# ~~~~~~~~~~ Distances, CBLOF core and labels redux ~~~~~~~~~~~~~~~~
assigned = centroid_scaled[labels]
dist_to_centroid = np.linalg.norm(num_features_scaled - assigned, axis=1)

rules_df["dist_to_centroid"] = dist_to_centroid

cluster_sizes = np.bincount(labels, minlength=cluster_num)

cblof = cluster_sizes[labels] * dist_to_centroid
rules_df["cblof"] = cblof

# ********* distance histogram ********
plt.figure()
plt.hist(rules_df["dist_to_centroid"], bins=50)
plt.xlabel("Distance to Centroid Histogram(scaled)")
plt.ylabel("Count")
plt.title("Distribution of Distances to Centroid")
plt.tight_layout()
plt.savefig("outputs/CBLOF_distance_hist.png", dpi=150)
plt.show()
# ***************************************

def label_anomalies_cblof(cblof_scores, method="highest_scores", highest_scores=22, percent=95, k=3.0):
    s = np.asarray(cblof_scores)

    if method == "highest_scores":
        highest_id = np.argsort(s)[-highest_scores:]
        flags = np.zeros_like(s, dtype=bool)
        flags[highest_id] = True
        threshold = s[highest_id].min() if highest_scores > 0 else np.inf
    elif method == "percent":
        threshold = np.percentile(s, percent)
        flags = s >= threshold
    elif method == "stat":
        mu, sigma = s.mean(), s.std(ddof=0)
        threshold = mu + k * sigma
        flags = s >= threshold
    else:
        raise ValueError("method must be 'highest_scores', 'percentile', or 'stat'")
    return flags, threshold

anomaly_flags, cblof_threshold = label_anomalies_cblof(
    rules_df["cblof"].values,
    method="highest_scores",
    highest_scores=22,
    percent=95,
    k=3.0
)

rules_df["is_anomalous"] = anomaly_flags

anomalies = rules_df[rules_df["is_anomalous"]]
normals = rules_df[~rules_df["is_anomalous"]]

feature_diffs = (
    anomalies[numeric_columns].mean() - normals[numeric_columns].mean()
).abs().sort_values(ascending=False)

# ********* Plot top feature differences *********
plt.figure(figsize=(8, 5))
feature_diffs.head(10).plot(kind='barh', color='firebrick', alpha=0.8)
plt.gca().invert_yaxis()
plt.xlabel("Mean Difference (Anomaly vs Normal)")
plt.title(f"Top Features Driving Anomalies")
plt.tight_layout()
plt.savefig("outputs/CBLOF_top_features_driving_anomalies.png", dpi=150)
plt.show()

# *** Threshold Histogram after Anomoly detection ***
plt.figure()
plt.hist(rules_df["cblof"], bins=50)
plt.axvline(cblof_threshold, color='r', linestyle="--")
plt.xlabel("CBLOF Score")
plt.ylabel("Count")
plt.title(f"CBLOF Distribution (threshold = {cblof_threshold:.2f})")
plt.tight_layout()
plt.savefig("outputs/CBLOF_hist.png", dpi=150)
plt.show()

# ********* Anomolies Scattrter Plot **************
pca = PCA(n_components=2, random_state=42)
points_2d = pca.fit_transform(num_features_scaled)
centroids_2d = pca.transform(centroid_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(points_2d[~rules_df["is_anomalous"], 0],
            points_2d[~rules_df["is_anomalous"], 1],
            color='blue', s=8, alpha=0.3, label='Normal')

plt.scatter(points_2d[rules_df["is_anomalous"], 0],
            points_2d[rules_df["is_anomalous"], 1],
            color='red', edgecolors='black', linewidths=0.3, s=18, alpha=0.85, label='Anomaly')

plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
            s=100, marker="x", linewidths=2, color="black", label="Centroid")

plt.title("Clusters with Anomalies Highlighted")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/CBLOF_clusters_normals_vs_anomalies.png", dpi=200)
plt.show()

rules_df.to_csv("outputs/rules_with_cblof.csv", index=False)
pd.DataFrame({"cluster": np.arange(cluster_num), "size": cluster_sizes}).to_csv(
    "outputs/kMean_cluster_sizes.csv", index=False
)

# ********* overall outcome ********
print()
print("--- Diagnostics ---")
print("Rows after cleaning:", len(rules_df))
print("points_2d shape:", points_2d.shape)
print("Anomalies:", rules_df["is_anomalous"].sum(),
      "Normals:", (~rules_df["is_anomalous"]).sum())
print("Unique clusters:", np.unique(labels))
print("Cluster sizes:", np.bincount(labels))
assert points_2d.shape[0] == len(rules_df), "Mismatch: points_2d vs rules_df rows"

# ********* TOP Features ********
top_features = feature_diffs.head(2).index.tolist()
plt.figure(figsize=(7, 5))
plt.scatter(rules_df.loc[~rules_df["is_anomalous"], top_features[0]], rules_df.loc[~rules_df["is_anomalous"], top_features[1]], color='blue', s=10, alpha=0.4, label='Normal')
plt.scatter(rules_df.loc[rules_df["is_anomalous"], top_features[0]], rules_df.loc[rules_df["is_anomalous"], top_features[1]], color='red', edgecolors='black', linewidths=0.3, s=25, alpha=0.9, label='Anomaly')
plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.title(f"Anomalies in Feature Space: {top_features[0]} vs {top_features[1]}")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/CBLOF_top_features_scatter.png", dpi=150)
plt.show()

# ********* 2D Cluster scatter plot ********
plt.figure()
plt.scatter(points_2d[:, 0], points_2d[:, 1], s=10, alpha=0.6, c=labels)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=120, marker="X", edgecolors="black")
plt.title("Clusters in 2D Space")
plt.tight_layout()
plt.savefig("outputs/CBLOF_clusters_2d.png", dpi=150)
plt.show()

# ********* 3D Cluster scatter plot ********
pca3d = PCA(n_components=3, random_state=42)
points_3d = pca3d.fit_transform(num_features_scaled)
centroids_3d = pca3d.transform(centroid_scaled)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points_3d[~rules_df["is_anomalous"], 0],
           points_3d[~rules_df["is_anomalous"], 1],
           points_3d[~rules_df["is_anomalous"], 2],
           color='blue', s=12, alpha=0.35, label='Normal')

ax.scatter(points_3d[rules_df["is_anomalous"], 0],
           points_3d[rules_df["is_anomalous"], 1],
           points_3d[rules_df["is_anomalous"], 2],
           color='red', edgecolors='black', linewidths=0.3, s=25, alpha=0.85, label='Anomaly')

ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2],
           color='black', marker='x', s=120, linewidths=2.2, label='Centroid')

ax.set_title("3D View of Clusters and Anomalies")
ax.legend(loc='best')
plt.tight_layout()
plt.savefig("outputs/CBLOF_clusters_3d.png", dpi=200)
plt.show()
# ***************************************

# ~~~~~~~~~~~~~ Building a clean CBLOF score table ~~~~~~~~~~~~~~~~~
id_cols = rules_df.select_dtypes(exclude=np.number).columns.tolist()
cluster_size_map = pd.Series(cluster_sizes, index=np.arange(cluster_num))
per_row_cluster_size = cluster_size_map.loc[labels].values

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

cblof_scores_df.to_csv(CBLOF_OUTPUT, index=False)

# ~~~~~~~~~~~~~~~~~~~~~~ Silhouette score ~~~~~~~~~~~~~~~~~~~~~~~~~~
try:
    sil = silhouette_score(num_features_scaled, labels)
except Exception:
    sil = float("nan")

print("CBLOF completed.")
print(f"Silhouette Score: {sil:.2f}")

print(f"Detected {actual_highest_scores} anomalous rules.")
print(f"Saved results to {CBLOF_OUTPUT}")
