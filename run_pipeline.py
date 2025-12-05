import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

scripts = [
    "pre_processing_one.py",
    "pre_procesing_two.py",
    "apriori.py",
    "vectorization.py",
    "cblof.py",
    "ecod.py",
    "isolation_forest.py",
    "one_class_svm.py"
]

print("=== Running Data Mining Pipeline ===")
for script in scripts:
    print(f"\n--- Running {script} ---")
    result = subprocess.run(["python3", script], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error running {script}:\n{result.stderr}")
        break
print("\n=== Pipeline Complete ===")

# --- Summary of Anomaly Detection Results ---
print("\n--- Anomaly Detection Summary ---")

# --- ECOD Summary ---
try:
    ecod_df = pd.read_csv('outputs/ecod_outliers.csv')
    ecod_anomalies = len(ecod_df)
    ecod_silhouette = ecod_df['Silhouette_Score'].iloc[0]
    print(f"ECOD: {ecod_anomalies} anomalies found with a silhouette score of {ecod_silhouette:.4f}")
except FileNotFoundError:
    print("ECOD: Results not found.")
except IndexError:
    print("ECOD: Could not read silhouette score from output file.")

# --- CBLOF Summary ---
try:
    cblof_df = pd.read_csv('outputs/cblof_outliers.csv')
    cblof_anomalies = len(cblof_df)
    cblof_silhouette = cblof_df['Silhouette_Score'].iloc[0]
    print(f"CBLOF: {cblof_anomalies} anomalies found with a silhouette score of {cblof_silhouette:.4f}")
except FileNotFoundError:
    print("CBLOF: Results not found.")
except IndexError:
    print("CBLOF: Could not read silhouette score from output file.")

# --- Isolation Forest Summary ---
try:
    iso_forest_df = pd.read_csv('outputs/isolation_forest_outliers.csv')
    iso_forest_anomalies = len(iso_forest_df)
    iso_forest_silhouette = iso_forest_df['Silhouette_Score'].iloc[0]
    print(f"Isolation Forest: {iso_forest_anomalies} anomalies found with a silhouette score of {iso_forest_silhouette:.4f}")
except FileNotFoundError:
    print("Isolation Forest: Results not found.")
except IndexError:
    print("Isolation Forest: Could not read silhouette score from output file.")


# --- One-Class SVM Summary ---
try:
    ocsvm_df = pd.read_csv('outputs/one_class_svm_outliers.csv')
    ocsvm_anomalies = len(ocsvm_df)
    ocsvm_silhouette = ocsvm_df['Silhouette_Score'].iloc[0]
    print(f"One-Class SVM: {ocsvm_anomalies} anomalies found with a silhouette score of {ocsvm_silhouette:.4f}")
except FileNotFoundError:
    print("One-Class SVM: Results not found.")
except IndexError:
    print("One-Class SVM: Could not read silhouette score from output file.")

# --- Common Anomaly Analysis ---
print("\n--- Common Anomaly Analysis ---")

# Dictionary to hold the sets of anomaly identifiers from each algorithm
anomaly_sets = {}
dataframes_to_load = {
    "CBLOF": 'outputs/cblof_outliers.csv',
    "ECOD": 'outputs/ecod_outliers.csv',
    "Isolation Forest": 'outputs/isolation_forest_outliers.csv',
    "One-Class SVM": 'outputs/one_class_svm_outliers.csv'
}

# Function to create a unique identifier for each rule
def create_rule_identifier(row):
    return f"{row['antecedents']} -> {row['consequents']}"

# Load each file and create a set of anomaly identifiers
for name, path in dataframes_to_load.items():
    try:
        df = pd.read_csv(path)
        if not df.empty:
            # Create a set of unique identifiers for the anomalies
            anomaly_sets[name] = set(df.apply(create_rule_identifier, axis=1))
    except FileNotFoundError:
        print(f"Note: {name} results file not found, skipping from common analysis.")
    except Exception as e:
        print(f"Note: Could not process {name} results file. Error: {e}")

# Only proceed if we have at least two sets to compare
if len(anomaly_sets) >= 2:
    from itertools import combinations

    # --- Find anomalies common to ALL available algorithms ---
    if len(anomaly_sets) > 1:
        common_all = set.intersection(*anomaly_sets.values())
        print(f"\nAnomalies common to ALL {len(anomaly_sets)} algorithms: {len(common_all)}")
        if 0 < len(common_all) <= 10:  # Print if the list is small
            print("  - " + "\n  - ".join(sorted(list(common_all))))

    # --- Find anomalies common to combinations of algorithms ---
    algorithm_names = list(anomaly_sets.keys())
    
    # Check for combinations of 3 (if there are at least 3 algos)
    if len(algorithm_names) >= 3:
        print("\nAnomalies common to combinations of 3 algorithms:")
        for combo in combinations(algorithm_names, 3):
            set1, set2, set3 = anomaly_sets[combo[0]], anomaly_sets[combo[1]], anomaly_sets[combo[2]]
            intersection = set1 & set2 & set3
            print(f"  - {combo[0]}, {combo[1]}, {combo[2]}: {len(intersection)} common anomalies")

    # Check for combinations of 2
    print("\nAnomalies common to combinations of 2 algorithms:")
    for combo in combinations(algorithm_names, 2):
        set1, set2 = anomaly_sets[combo[0]], anomaly_sets[combo[1]]
        intersection = set1 & set2
        print(f"  - {combo[0]}, {combo[1]}: {len(intersection)} common anomalies")

else:
    print("Could not perform common anomaly analysis: Less than two result files were found.")
