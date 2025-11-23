import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load the association rules generated in Stage 1
rules = pd.read_csv('outputs/rules.csv')

print(f"✓ Loaded {len(rules)} rows from rules.csv")


features_to_map = ['antecedents', 'consequents', 'antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'leverage', 'zhangs_metric', 'jaccard', 'certainty', 'kulczynski']
unscaled_data = rules[features_to_map]

features_to_scale = ['antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'leverage', 'zhangs_metric', 'jaccard', 'certainty', 'kulczynski']
data_to_scale = rules[features_to_scale]

# Save the unscaled data to a CSV file for later mapping
unscaled_data.to_csv('outputs/unscaled_vectors.csv', index=False)
print(f"✓ Saved unscaled vectors to outputs/unscaled_vectors.csv")

# Scale the features to have mean=0 and variance=1
vector = StandardScaler().fit_transform(data_to_scale)

print(f"✓ Loaded {len(vector)} vectors with {vector.shape[1]} dimensions each.")

# Save the scaled vectors to a CSV file
vector_df = pd.DataFrame(vector, columns=features_to_scale)
vector_df.to_csv('outputs/vectorized.csv', index=False)
print(f"✓ Saved scaled vectors to outputs/vectorized.csv")