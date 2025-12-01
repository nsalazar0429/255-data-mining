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
CBLOF_OUTPUT = "outputs/cblof_outliers.csv"

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

cluster_num = best_k

RANDOM_STATE = 42   #default number used for random states

try:
    kMean_model = KMeans(n_clusters=cluster_num, random_state=RANDOM_STATE, n_init="auto")
except TypeError:
    kMean_model = KMeans(n_clusters=cluster_num, random_state=RANDOM_STATE, n_init=10)

labels = kMean_model.fit_predict(num_features_scaled)

centroid_scaled = kMean_model.cluster_centers_
centroid_original = scaler.inverse_transform(centroid_scaled)

cluster_size = np.bincount(labels, minlength=cluster_num)

centroids_df = pd.DataFrame(centroid_original, columns=numeric_columns)
centroids_df.index.name = "Cluster"

rules_df["kMean_label"] = labels

# ~~~~~~~~~~ Distances, CBLOF core and labels redux ~~~~~~~~~~~~~~~~
assigned = centroid_scaled[labels]
dist_to_centroid = np.linalg.norm(num_features_scaled - assigned, axis=1)

rules_df["dist_to_centroid"] = dist_to_centroid

cluster_sizes = np.bincount(labels, minlength=cluster_num)

cblof = cluster_sizes[labels] * dist_to_centroid
rules_df["cblof"] = cblof

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

# ********* overall outcome ********
print()
print("--- Diagnostics ---")
print("Total Data point Rules:", len(rules_df))
print("Number of Rule Anomalies:", rules_df["is_anomalous"].sum(),
      "\nNumber of Normal Rules:", (~rules_df["is_anomalous"]).sum())
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
cblof_scores = rules_df["cblof"].values

sorted_idx = np.argsort(cblof_scores) 

descending_idx = []
for i in range(len(sorted_idx) - 1, -1, -1):
    descending_idx.append(sorted_idx[i])

descending_idx = np.array(descending_idx)

if len(cblof_scores) >= 4:
    anom_idx = descending_idx[0:4]
else:
    anom_idx = descending_idx

source_df = rules_df.iloc[anom_idx].copy()

required_rule_cols = [
    "antecedents", "consequents",
    "antecedent support", "consequent support",
    "support", "confidence", "lift", "leverage",
    "zhangs_metric", "jaccard", "certainty", "kulczynski"
]

for col in required_rule_cols:
    if col not in source_df.columns:
        source_df[col] = np.nan

cblof_scores_df = source_df[required_rule_cols].copy()

cblof_scores_df["IF_Anomaly"] = -1 
cblof_scores_df["IF_Score"] = cblof_scores[anom_idx]

try:
    sil = silhouette_score(num_features_scaled, labels)
except Exception:
    sil = float("nan")

cblof_scores_df["Silhouette_Score"] = sil

contamination = len(anom_idx) / len(rules_df) if len(rules_df) > 0 else float("nan")
cblof_scores_df["Contamination"] = contamination

cblof_scores_df["N_estimators"] = 10

os.makedirs(os.path.dirname(CBLOF_OUTPUT), exist_ok=True)

topFour = 4
actual_highest_scores = len(cblof_scores_df)
if actual_highest_scores != topFour:
    print(f"Warning: expected {topFour} anomalies in output, found {actual_highest_scores}")

cblof_scores_df.to_csv(CBLOF_OUTPUT, index=False)

# ~~~~~~~~~~~~~~~~~~~~~~ Silhouette score ~~~~~~~~~~~~~~~~~~~~~~~~~~
try:
    sil = silhouette_score(num_features_scaled, labels)
except Exception:
    sil = float("nan")

print("\nCBLOF completed.")
print(f"Best Silhouette Score: {best_sil:.3f}")

print(f"\nSaved results to {CBLOF_OUTPUT}")
