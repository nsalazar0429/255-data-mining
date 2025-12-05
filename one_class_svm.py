"""
One-Class SVM for Anomaly Detection
====================================
This script uses One-Class SVM to spot unusual insurance claims that might be fraudulent.
Think of it like a smart detective: it learns what "normal" claims look like, then flags anything that seems out of place.

Author: Mahalakshmi R
Date: November 23, 2025
Course: CMPE 255 - Data Mining
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt

#  set up some options for the model
NU_RANGE = [round(i * 0.001, 3) for i in range(1, 25)]  
GAMMA_OPTIONS = ['scale', 'auto'] + [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
KERNEL_OPTIONS = ['rbf', 'poly', 'sigmoid']
POLY_DEGREES = [2, 3, 4]

PERCENTILE_THRESHOLD = 99

UNSCALED_DATA_PATH = 'outputs/unscaled_vectors.csv'
SCALED_DATA_PATH = 'outputs/vectorized.csv'
OUTPUT_GRAPH_PATH = 'outputs/top_drivers_3d_ONE_CLASS_SVM.png'
OUTPUT_OUTLIERS_PATH = 'outputs/one_class_svm_outliers.csv'

def tune_svm_parameters(scaled_vector_values, nu_range, kernel_options, gamma_options):
    """This function tries different settings to find the best fit."""
    best_score = -1
    best_params = {}
    
    print("Trying out different One-Class SVM settings...")

    for nu in nu_range:
        for kernel in kernel_options:
            for gamma in gamma_options:
                try:
                    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
                    preds = model.fit_predict(scaled_vector_values)
                    
                    if len(set(preds)) > 1:
                        score = silhouette_score(scaled_vector_values, preds)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'nu': nu, 'kernel': kernel, 'gamma': gamma}
                except:
                    continue
    
    return best_params, best_score

def tune_polynomial_kernel(scaled_vector_values, nu_range, gamma_options, degrees):
    """Similar to above but with the polynomial kernel with different degrees."""
    best_score = -1
    best_params = {}
    
    print("Testing polynomial kernel with different degrees...")

    for nu in nu_range:
        for gamma in gamma_options:
            for degree in degrees:
                try:
                    model = OneClassSVM(nu=nu, kernel='poly', gamma=gamma, degree=degree)
                    preds = model.fit_predict(scaled_vector_values)
                    
                    if len(set(preds)) > 1:
                        score = silhouette_score(scaled_vector_values, preds)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'nu': nu, 'gamma': gamma, 'degree': degree}
                except:
                    continue
    
    return best_params, best_score

def apply_percentile_thresholding(scores, percentile=99):
    """Flag the top X% as outliers based on their scores."""
    threshold = np.percentile(scores, percentile)
    labels = np.where(scores >= threshold, -1, 1)
    return labels, threshold

def identify_top_driver_features(scaled_vector_df, anomaly_labels, feature_cols):
    """Find which features stand out the most in the outliers."""
    outlier_rows = scaled_vector_df[anomaly_labels == -1][feature_cols]
    
    outlier_means = outlier_rows.mean()
    outlier_top_features = {}
    for index, row in outlier_rows.iterrows():
        top_3 = row.abs().nlargest(3).index.tolist()
        outlier_top_features[index] = top_3
    
    # Big differences mean those features are more important
    all_features = [feature for features in outlier_top_features.values() for feature in features]
    top_drivers = [feature for feature, _ in Counter(all_features).most_common(3)]
    return top_drivers

def identify_top_driver_features_frequency(scaled_vector_df, anomaly_labels, feature_cols):
    """Find top features by counting how often they show up in outliers."""
    outlier_rows = scaled_vector_df[anomaly_labels == -1][feature_cols]
    
    # For each outlier, select its top 3 features
    outlier_top_features = {}
    for index, row in outlier_rows.iterrows():
        top_3 = row.abs().nlargest(3).index.tolist()
        outlier_top_features[index] = top_3
    
    # Count how many times each feature appears
    all_features = [feature for features in outlier_top_features.values() for feature in features]
    top_drivers = [feature for feature, _ in Counter(all_features).most_common(3)]
    
    return top_drivers

def create_3d_visualization(unscaled_vector_df, anomaly_labels, top_drivers, output_path):
    """Make a cool 3D plot to see outliers vs. normal points."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x_col, y_col, z_col = top_drivers
    
    # Show normal points
    normal_indices = unscaled_vector_df[anomaly_labels == 1].index
    ax.scatter(unscaled_vector_df.loc[normal_indices, x_col],
               unscaled_vector_df.loc[normal_indices, y_col],
               unscaled_vector_df.loc[normal_indices, z_col],
               c='lightblue', label='Normal', alpha=0.5, s=20)
    
    # Highlight outliers
    outlier_indices = unscaled_vector_df[anomaly_labels == -1].index
    ax.scatter(unscaled_vector_df.loc[outlier_indices, x_col],
               unscaled_vector_df.loc[outlier_indices, y_col],
               unscaled_vector_df.loc[outlier_indices, z_col],
               c='red', label='Outliers', s=100, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
    ax.set_zlabel(z_col, fontsize=11, fontweight='bold')
    ax.set_title('3D Visualization of Top Driver Features - One-Class SVM', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_comparison_table(rbf_counts, poly_counts, rbf_score, poly_score, total_samples):
    """Show a nice table comparing RBF and Polynomial results."""
    print("\n" + "=" * 60)
    print("COMPARISON: RBF vs Polynomial Kernel")
    print("=" * 60)
    print(f"{'Kernel':<20} {'Normal':<15} {'Anomaly':<15} {'Normal %':<15} {'Anomaly %':<15} {'Silhouette':<15}")
    print("-" * 95)
    
    rbf_normal = rbf_counts.get(1, 0)
    rbf_anomaly = rbf_counts.get(-1, 0)
    poly_normal = poly_counts.get(1, 0)
    poly_anomaly = poly_counts.get(-1, 0)
    
    print(f"{'RBF':<20} {rbf_normal:<15} {rbf_anomaly:<15} "
          f"{rbf_normal/total_samples*100:<15.1f} {rbf_anomaly/total_samples*100:<15.1f} {rbf_score:<15.4f}")
    print(f"{'Polynomial':<20} {poly_normal:<15} {poly_anomaly:<15} "
          f"{poly_normal/total_samples*100:<15.1f} {poly_anomaly/total_samples*100:<15.1f} {poly_score:<15.4f}")
    print(f"\nNote: Using percentile-based thresholding (bottom {100 - PERCENTILE_THRESHOLD}% as outliers)")

def plot_top_10_feature_diffs_ocsvm(scaled_vector_df):

    # Select only numeric feature columns (excluding anomaly columns)
    numeric_cols = [col for col in scaled_vector_df.columns if col not in ['SVM_Anomaly', 'SVM_RBF_Anomaly', 'SVM_Poly_Anomaly']]

    # Split outliers and normals
    anomalies = scaled_vector_df[scaled_vector_df['SVM_Anomaly'] == -1]
    normals = scaled_vector_df[scaled_vector_df['SVM_Anomaly'] == 1]

    # Calculate mean difference for each feature
    feature_diffs = (anomalies[numeric_cols].mean() - normals[numeric_cols].mean()).abs().sort_values(ascending=False)

    # Plot top 10 feature differences
    plt.figure(figsize=(8, 5))
    feature_diffs.head(10).plot(kind='barh', color='firebrick', alpha=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Difference (Anomaly vs Normal)")
    plt.title("Top Features Driving SVM Anomalies")
    plt.tight_layout()
    plt.savefig("outputs/OCSVM_top_features_driving_anomalies.png", dpi=150)
    plt.close()
    print("✓ Saved driving features plot to outputs/OCSVM_top_features_driving_anomalies.png")

def main():
    """Run the whole analysis step by step."""
    print("=" * 60)
    print("One-Class SVM for Anomaly Detection")
    print("=" * 60)
    
    # Load the data
    print("\n[1/6] Loading data...")
    unscaled_vector_df = pd.read_csv(UNSCALED_DATA_PATH)
    scaled_vector_df = pd.read_csv(SCALED_DATA_PATH)
    scaled_vector_values = scaled_vector_df.values
    
    print(f"✓ Loaded {len(unscaled_vector_df)} samples with {len(scaled_vector_df.columns)} features")
    
    # Find the best settings
    print("\n[2/6] Testing different model settings...")
    best_params, best_score = tune_svm_parameters(
        scaled_vector_values, NU_RANGE, KERNEL_OPTIONS, GAMMA_OPTIONS
    )
    print(f"✓ Best settings: {best_params}")
    print(f"✓ Silhouette Score: {best_score:.4f}")
    
    # Try the RBF kernel
    print("\n[3/6] Running One-Class SVM with RBF Kernel...")
    print("-" * 60)
    svm_rbf = OneClassSVM(nu=best_params['nu'], kernel='rbf', gamma=best_params['gamma'])
    svm_rbf.fit(scaled_vector_values)
    
    unscaled_vector_df['SVM_RBF_Score'] = -svm_rbf.decision_function(scaled_vector_values)
    scaled_vector_df['SVM_RBF_Score'] = unscaled_vector_df['SVM_RBF_Score']
    
    svm_rbf_labels, _ = apply_percentile_thresholding(
        unscaled_vector_df['SVM_RBF_Score'], PERCENTILE_THRESHOLD
    )
    unscaled_vector_df['SVM_RBF_Anomaly'] = svm_rbf_labels
    scaled_vector_df['SVM_RBF_Anomaly'] = svm_rbf_labels
    
    svm_rbf_counts = pd.Series(svm_rbf_labels).value_counts().to_dict()
    print(f"✓ RBF Kernel Results: {svm_rbf_counts}")
    
    # Try the Polynomial kernel
    print("\n[4/6] Running One-Class SVM with Polynomial Kernel...")
    print("-" * 60)
    best_poly_params, best_poly_score = tune_polynomial_kernel(
        scaled_vector_values, NU_RANGE, GAMMA_OPTIONS, POLY_DEGREES
    )
    print(f"✓ Best Poly settings: {best_poly_params}")
    print(f"✓ Silhouette Score: {best_poly_score:.4f}")
    
    svm_poly = OneClassSVM(
        nu=best_poly_params['nu'],
        kernel='poly',
        gamma=best_poly_params['gamma'],
        degree=best_poly_params['degree']
    )
    svm_poly.fit(scaled_vector_values)
    
    unscaled_vector_df['SVM_Poly_Score'] = -svm_poly.decision_function(scaled_vector_values)
    scaled_vector_df['SVM_Poly_Score'] = unscaled_vector_df['SVM_Poly_Score']
    
    svm_poly_labels, _ = apply_percentile_thresholding(
        unscaled_vector_df['SVM_Poly_Score'], PERCENTILE_THRESHOLD
    )
    unscaled_vector_df['SVM_Poly_Anomaly'] = svm_poly_labels
    scaled_vector_df['SVM_Poly_Anomaly'] = svm_poly_labels
    
    svm_poly_counts = pd.Series(svm_poly_labels).value_counts().to_dict()
    print(f"✓ Polynomial Kernel Results: {svm_poly_counts}")
    
    # Pick the main results
    unscaled_vector_df['SVM_Anomaly'] = unscaled_vector_df['SVM_RBF_Anomaly']
    unscaled_vector_df['SVM_Score'] = unscaled_vector_df['SVM_RBF_Score']
    scaled_vector_df['SVM_Anomaly'] = unscaled_vector_df['SVM_RBF_Anomaly']
    scaled_vector_df['SVM_Score'] = unscaled_vector_df['SVM_RBF_Score']
    
    # Show the comparison
    print_comparison_table(
        svm_rbf_counts, svm_poly_counts, best_score, best_poly_score, len(unscaled_vector_df)
    )
    
    # Decide which kernel is better based on silhouette score
    kernel_scores = {
        "rbf": best_score,
        "poly": best_poly_score
    }
    best_kernel = max(kernel_scores, key=kernel_scores.get)
    print(f"\n✓ Best kernel based on silhouette score: {best_kernel}")

    # Map kernel name to the correct anomaly column and parameters
    kernel_to_anom_col = {
        "rbf": "SVM_RBF_Anomaly",
        "poly": "SVM_Poly_Anomaly"
    }
    kernel_to_params = {
        "rbf": {"params": best_params, "score": best_score},
        "poly": {"params": best_poly_params, "score": best_poly_score}
    }

    anom_col = kernel_to_anom_col[best_kernel]
    y_anom = unscaled_vector_df[anom_col]
    best_kernel_info = kernel_to_params[best_kernel]
    
    # Save outlier rows with original values, Silhouette scores, and best parameters
    outliers = unscaled_vector_df[y_anom == -1].copy()
    if len(outliers) > 0:
        # Drop the old main columns (created at lines 241-242) to avoid duplicates
        if 'SVM_Anomaly' in outliers.columns:
            outliers = outliers.drop(columns=['SVM_Anomaly'])
        if 'SVM_Score' in outliers.columns:
            outliers = outliers.drop(columns=['SVM_Score'])
        
        # Only keep feature columns and the best kernel's anomaly/score columns (like IF does)
        # Remove other SVM columns to match IF structure
        if best_kernel == "rbf":
            # Rename best kernel columns to match IF naming pattern
            outliers = outliers.rename(columns={'SVM_RBF_Anomaly': 'SVM_Anomaly', 'SVM_RBF_Score': 'SVM_Score'})
            # Drop other kernel columns
            cols_to_drop = ['SVM_Poly_Anomaly', 'SVM_Poly_Score']
        else:  # poly
            # Rename best kernel columns to match IF naming pattern
            outliers = outliers.rename(columns={'SVM_Poly_Anomaly': 'SVM_Anomaly', 'SVM_Poly_Score': 'SVM_Score'})
            # Drop other kernel columns
            cols_to_drop = ['SVM_RBF_Anomaly', 'SVM_RBF_Score']
        
        outliers = outliers.drop(columns=[col for col in cols_to_drop if col in outliers.columns])
        
        # Reorder columns to match IF format: features, then SVM_Anomaly, SVM_Score, then parameters
        feature_cols = [col for col in outliers.columns if col not in ['SVM_Anomaly', 'SVM_Score', 'Silhouette_Score', 'Nu', 'Gamma', 'Kernel', 'Degree']]
        other_cols = ['SVM_Anomaly', 'SVM_Score'] if 'SVM_Anomaly' in outliers.columns and 'SVM_Score' in outliers.columns else []
        param_cols = ['Silhouette_Score', 'Nu', 'Gamma']
        if 'Degree' in outliers.columns:
            param_cols.append('Degree')
        param_cols.append('Kernel')
        outliers = outliers[[col for col in feature_cols + other_cols + param_cols if col in outliers.columns]]
        
        # Add parameters (similar to IF's Contamination and N_estimators)
        outliers['Silhouette_Score'] = best_kernel_info["score"]
        if best_kernel == "rbf":
            outliers['Nu'] = best_params['nu']
            outliers['Gamma'] = best_params['gamma']
            outliers['Kernel'] = 'rbf'
        else:  # poly
            outliers['Nu'] = best_poly_params['nu']
            outliers['Gamma'] = best_poly_params['gamma']
            outliers['Degree'] = best_poly_params['degree']
            outliers['Kernel'] = 'poly'
        
        outliers.to_csv(OUTPUT_OUTLIERS_PATH, index=False)
        print(f"✓ Saved outliers with original values, Silhouette score, and best parameters to {OUTPUT_OUTLIERS_PATH}")
    
    # Find the most important features using the BEST kernel
    print("\n[5/6] Figuring out which features matter most (best kernel)...")
    print("-" * 60)

    feature_cols = [
        col for col in scaled_vector_df.columns
        if col not in [
            'SVM_Anomaly', 'SVM_Score',
            'SVM_RBF_Anomaly', 'SVM_RBF_Score',
            'SVM_Poly_Anomaly', 'SVM_Poly_Score'
        ]
    ]

    if len(unscaled_vector_df[y_anom == -1]) > 0:
        # Use two methods to find top features
        top_drivers_statistical = identify_top_driver_features(
            scaled_vector_df, y_anom, feature_cols
        )
        print(f"✓ Statistical Method (Mean Difference): {top_drivers_statistical}")

        top_drivers_frequency = identify_top_driver_features_frequency(
            scaled_vector_df, y_anom, feature_cols
        )
        print(f"✓ Frequency-Based Method (Top-3 Count): {top_drivers_frequency}")

        # If both methods agree, use that. Otherwise, go with the statistical one.
        if top_drivers_statistical == top_drivers_frequency:
            print(f"✓ Both methods agree! Using: {top_drivers_statistical}")
            top_drivers = top_drivers_statistical
        else:
            print("⚠ Methods differ. Using statistical method for visualization.")
            top_drivers = top_drivers_statistical

        # Make a 3D plot with best kernel’s anomalies
        print("\n[6/6] Creating a 3D visualization...")
        create_3d_visualization(
            unscaled_vector_df, y_anom, top_drivers, OUTPUT_GRAPH_PATH
        )
        print(f"✓ 3D scatter plot saved to {OUTPUT_GRAPH_PATH}")
    else:
        print("⚠ No outliers found for the best kernel, so skipping the visualization")

    plot_top_10_feature_diffs_ocsvm(scaled_vector_df)

    print("\n" + "=" * 60)
    print("✓ All done! One-Class SVM analysis finished.")
    print("=" * 60)


    
    return unscaled_vector_df, scaled_vector_df

if __name__ == "__main__":
    main()
