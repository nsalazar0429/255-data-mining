import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt

contamination_range = [round(i * 0.01, 2) for i in range(1, 21)]

# Load the association rules generated in Stage 1``
vector_scaled = pd.read_csv('outputs/vectorized.csv')
# Load the original unscaled data for reference
unscaled_vector_df = pd.read_csv('outputs/unscaled_vectors.csv')

# Set up variables to track the best model and its score
best_ecod_score = -1
best_ecod_params = {}

print("Tuning ECOD parameters...")

for cont in contamination_range:
    # Initialize and fit
    model = ECOD(contamination=cont)
    preds = model.fit(vector_scaled.values).predict(vector_scaled.values)

    # Calculate Score (Only if more than 1 cluster exists)
    if len(set(preds)) > 1:
        score = silhouette_score(vector_scaled, preds)

        if score > best_ecod_score:
            best_ecod_score = score
            best_ecod_params = {'contamination': cont}

print(f"✓ Best params: {best_ecod_params}")
print(f"✓ Silhouette Score: {best_ecod_score:.4f}")

# Initialize and fit the ECOD model
ecod = ECOD(contamination=best_ecod_params['contamination'])

# Map ECOD anomaly results: 0 -> 1 (inlier), 1 -> -1 (outlier)
ec_pred = preds = ecod.fit(vector_scaled.values).predict(vector_scaled.values)
vector_scaled['ECOD_Anomaly'] = [1 if x == 0 else -1 for x in ec_pred]
unscaled_vector_df['ECOD_Anomaly'] = [1 if x == 0 else -1 for x in ec_pred]

# Calculate the Silhouette Score
ecod_silhouette_score = silhouette_score(vector_scaled, ec_pred)

# Print the results
print(f"✓ ECOD Anomaly Counts (1=Normal, -1=Fraud): {unscaled_vector_df['ECOD_Anomaly'].value_counts().to_dict()}")

# Save outlier rows with original values, Silhouette score, and best parameters in separate columns
outliers = unscaled_vector_df[unscaled_vector_df['ECOD_Anomaly'] == -1].copy()
outliers['Silhouette_Score'] = best_ecod_score
outliers['Contamination'] = best_ecod_params['contamination']
outliers.to_csv('outputs/ecod_outliers.csv', index=False)
print(f"✓ Saved outliers with original values, Silhouette score, and best parameters to outputs/ecod_outliers.csv")

# --- 4. Identify the "Global Driver" Features ---
print("Identifying Top Driving Features...")

# Get outlier rows as a DataFrame first
outlier_rows_df = vector_scaled[vector_scaled['ECOD_Anomaly'] == -1]

# Select only the feature columns (exclude IF_Anomaly and IF_Score)
feature_cols = [col for col in vector_scaled.columns if col not in ['ECOD_Anomaly']]
outlier_rows = outlier_rows_df[feature_cols]

outlier_top_features = {}
for index, row in outlier_rows.iterrows():
    top_3 = row.abs().nlargest(3).index.tolist()
    outlier_top_features[index] = top_3

# Count the frequency of each feature in outlier_top_features
all_features = [feature for features in outlier_top_features.values() for feature in features]
top_drivers = [feature for feature, _ in Counter(all_features).most_common(3)]
print(f"✓ Top driver features: {top_drivers}")


# 3D scatter plot of the top 3 driver features, highlighting outliers
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x_col, y_col, z_col = top_drivers

# Plot all points
ax.scatter(unscaled_vector_df[x_col], unscaled_vector_df[y_col], unscaled_vector_df[z_col], c='lightblue', label='Normal', alpha=0.5)

# Highlight outlier points
outlier_indices = unscaled_vector_df[unscaled_vector_df['ECOD_Anomaly'] == -1].index
ax.scatter(unscaled_vector_df.loc[outlier_indices, x_col],
           unscaled_vector_df.loc[outlier_indices, y_col],
           unscaled_vector_df.loc[outlier_indices, z_col],
           c='red', label='Outliers', s=60, edgecolors='k')

ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_zlabel(z_col)
ax.set_title('3D Plot of Top Driver Features - ECOD')
ax.legend()
plt.savefig('outputs/top_drivers_3d_ECOD.png')
plt.close()
print('✓ 3D scatter plot saved to outputs/top_drivers_isolation_ECOD.png')

# --- Plot Top 10 Feature Differences for ECOD Anomalies ---
print("Plotting top feature differences for ECOD anomalies...")

# Select only numeric feature columns (excluding anomaly columns)
numeric_cols = [col for col in vector_scaled.columns if col not in ['ECOD_Anomaly']]

# Split outliers and normals
anomalies = vector_scaled[vector_scaled['ECOD_Anomaly'] == -1]
normals = vector_scaled[vector_scaled['ECOD_Anomaly'] == 1]

# Calculate mean difference for each feature
feature_diffs = (anomalies[numeric_cols].mean() - normals[numeric_cols].mean()).abs().sort_values(ascending=False)

# Plot top 10 feature differences
plt.figure(figsize=(8, 5))
feature_diffs.head(10).plot(kind='barh', color='firebrick', alpha=0.8)
plt.gca().invert_yaxis()
plt.xlabel("Mean Difference (Anomaly vs Normal)")
plt.title("Top Features Driving ECOD Anomalies")
plt.tight_layout()
plt.savefig("outputs/ECOD_top_features_driving_anomalies.png", dpi=150)
plt.close()
print("✓ Saved driving features plot to outputs/ECOD_top_features_driving_anomalies.png")

