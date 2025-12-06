import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt

# 1. Define the dynamic ranges
# Contamination: 0.01 to 0.20 (inclusive)
# We use range(1, 21) to get integers 1..20, then divide by 100
contamination_range = [round(i * 0.01, 2) for i in range(1, 21)]

# Estimators: 10 to 200, stepping by 10
# range(start, stop, step) -> stop is exclusive, so we use 210 to include 200
estimator_range = range(10, 210, 10)

# Load the original unscaled data for reference
unscaled_vector_df = pd.read_csv('outputs/unscaled_vectors.csv')

# Load the vectorized data for anomaly detection
scaled_vector_df = pd.read_csv('outputs/vectorized.csv')
scaled_vector_values = scaled_vector_df.values  # NumPy array of vectors
feature_means = scaled_vector_df.mean()

# Set up variables to track the best model and its score
best_if_score = -1
best_if_params = {}

print("Tuning Isolation Forest parameters...")

for cont in contamination_range:
    for est in estimator_range:
        # Initialize and fit
        model = IsolationForest(n_estimators=est, contamination=cont, random_state=42)
        preds = model.fit_predict(scaled_vector_values)

        # Calculate Score (Only if more than 1 cluster exists)
        if len(set(preds)) > 1:
            score = silhouette_score(scaled_vector_values, preds)

            if score > best_if_score:
                best_if_score = score
                best_if_params = {'contamination': cont, 'n_estimators': est}

print(f"✓ Best params: {best_if_params}")
print(f"✓ Silhouette Score: {best_if_score:.4f}")

# --- Final Model with Best Parameters ---
iso_forest = IsolationForest(n_estimators=best_if_params['n_estimators'], contamination=best_if_params['contamination'], random_state=42)

# Fit the model and predict labels: -1 for outliers, 1 for normal observations
unscaled_vector_df['IF_Anomaly'] = iso_forest.fit_predict(scaled_vector_values)
scaled_vector_df['IF_Anomaly'] = iso_forest.fit_predict(scaled_vector_values)

# Calculate the anomaly score: Invert it so higher scores indicate higher likelihood of fraud
unscaled_vector_df['IF_Score'] = iso_forest.decision_function(scaled_vector_values)
scaled_vector_df['IF_Score'] = iso_forest.decision_function(scaled_vector_values)

# Save the results with anomalies and scores
print(f"✓ Isolation Forest ( 1 = normal, -1 = anomaly ): {unscaled_vector_df['IF_Anomaly'].value_counts().to_dict()}")

# Save outlier rows with original values, Silhouette score, and best parameters in separate columns
outliers = unscaled_vector_df[unscaled_vector_df['IF_Anomaly'] == -1].copy()
outliers['Silhouette_Score'] = best_if_score
outliers['Contamination'] = best_if_params['contamination']
outliers['N_estimators'] = best_if_params['n_estimators']
outliers.to_csv('outputs/isolation_forest_outliers.csv', index=False)
print(f"✓ Saved outliers with original values, Silhouette score, and best parameters to outputs/isolation_forest_outliers.csv")

# --- Identify the "Global Driver" Features ---
print("Identifying Top Driving Features...")

# Get outlier rows as a DataFrame first
outlier_rows_df = scaled_vector_df[scaled_vector_df['IF_Anomaly'] == -1]

# Select only the feature columns (exclude IF_Anomaly and IF_Score)
feature_cols = [col for col in scaled_vector_df.columns if col not in ['IF_Anomaly', 'IF_Score']]
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
outlier_indices = unscaled_vector_df[unscaled_vector_df['IF_Anomaly'] == -1].index
ax.scatter(unscaled_vector_df.loc[outlier_indices, x_col],
           unscaled_vector_df.loc[outlier_indices, y_col],
           unscaled_vector_df.loc[outlier_indices, z_col],
           c='red', label='Outliers', s=60, edgecolors='k')

ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_zlabel(z_col)
ax.set_title('3D Plot of Top Driver Features - ISOLATION FOREST')
ax.legend()
plt.savefig('outputs/top_drivers_3d_ISOLATION_FOREST.png')
plt.close()
print('✓ 3D scatter plot saved to outputs/top_drivers_ISOLATION_FOREST.png')

# --- 5. Plot Top Feature Differences for Anomalies ---
print("Plotting top feature differences for anomalies...")

# Calculate mean difference between outliers and normals for each feature
normal_rows = scaled_vector_df[scaled_vector_df['IF_Anomaly'] == 1]
anomaly_rows = scaled_vector_df[scaled_vector_df['IF_Anomaly'] == -1]

feature_diff = (anomaly_rows[feature_cols].mean() - normal_rows[feature_cols].mean()).abs().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
feature_diff.head(10).plot(kind='barh', color='firebrick', alpha=0.8)
plt.gca().invert_yaxis()
plt.xlabel("Mean Difference (Anomaly vs Normal)")
plt.title("Top Features Driving Isolation Forest Anomalies")
plt.tight_layout()
plt.savefig("outputs/IF_top_features_driving_anomalies.png", dpi=150)
plt.close()
print("✓ Saved driving features plot to outputs/IF_top_features_driving_anomalies.png")