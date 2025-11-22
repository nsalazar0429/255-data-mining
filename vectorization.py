import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load the association rules generated in Stage 1
df = pd.read_csv('outputs/rules.csv')

print(f"✓ Loaded {len(df)} rows from rules.csv")
print(f"✓ Columns: {list(df.columns)}")

features = ['antecedent support', 'consequent support', 'support', 'confidence', 'lift', 
            'representativity', 'leverage', 'zhangs_metric', 'jaccard', 'certainty', 'kulczynski']
X = df[features]

# Scale the features to have mean=0 and variance=1
X_scaled = StandardScaler().fit_transform(X)

print(f"✓ Loaded {len(X_scaled)} vectors with {X_scaled.shape[1]} dimensions each.")

# Initialize the model with contamination=0.05 (expecting 5% outliers)
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

# Fit the model and predict labels: -1 for outliers, 1 for normal observations
df['IF_Anomaly'] = iso_forest.fit_predict(X_scaled)

# Calculate the anomaly score: Invert it so higher scores indicate higher likelihood of fraud
df['IF_Score'] = -iso_forest.decision_function(X_scaled)

print(f"✓ Isolation Forest ( 1 = normal, -1 = anomaly ): {df['IF_Anomaly'].value_counts().to_dict()}")

# Initialize OCSVM with RBF kernel and nu=0.05 (allowing ~5% outliers)
oc_svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')

# Fit the model and predict labels: -1 for outliers, 1 for normal observations
df['OCSVM_Anomaly'] = oc_svm.fit_predict(X_scaled)

print(f"✓ OCSVM ( 1 = normal, -1 = anomaly ): {df['OCSVM_Anomaly'].value_counts().to_dict()}")

# Calculate Silhouette Score: Higher is better (1.0 is perfect separation)
score_if = silhouette_score(X_scaled, df['IF_Anomaly'])
score_ocsvm = silhouette_score(X_scaled, df['OCSVM_Anomaly'])

print(f"✓ Isolation Forest Silhouette Score: {score_if:.3f}")
print(f"✓ OCSVM Silhouette Score: {score_ocsvm:.3f}")

# --- 1. Isolation Forest Outliers ---
print("\n--- Isolation Forest Anomalies (By Score) ---")
# Filter: Select rows where IF_Anomaly is -1
if_outliers = df[df['IF_Anomaly'] == -1].copy()

# Sort: Highest Anomaly Score first
if_outliers = if_outliers.sort_values(by='IF_Score', ascending=False)

# Display specific columns
cols = ['antecedents', 'consequents', 'IF_Score']
print(if_outliers[cols].head(5))


# --- 2. One-Class SVM Outliers ---
print("\n--- One-Class SVM Anomalies (By Lift) ---")
# Filter: Select rows where OCSVM_Anomaly is -1
ocsvm_outliers = df[df['OCSVM_Anomaly'] == -1].copy()

# Sort: Highest Lift first (since OCSVM doesn't give a simple "score" like IF)
ocsvm_outliers = ocsvm_outliers.sort_values(by='lift', ascending=False)

# Display specific columns
cols = ['antecedents', 'consequents', 'lift']
print(ocsvm_outliers[cols])


# --- 3. The "Consensus" List (High Priority) ---
print("\n--- High Priority: Rules flagged by BOTH models ---")
# Filter: Where BOTH are -1
consensus_rules = df[(df['IF_Anomaly'] == -1) & (df['OCSVM_Anomaly'] == -1)]