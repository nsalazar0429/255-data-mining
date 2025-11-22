import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load the association rules generated in Stage 1
df = pd.read_csv('outputs/rules.csv')

print(f"✓ Loaded {len(df)} rows from rules.csv")

features = ['antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'leverage', 'zhangs_metric', 'jaccard', 'certainty', 'kulczynski']
unscaled_data = df[features]

# Scale the features to have mean=0 and variance=1
vector = StandardScaler().fit_transform(unscaled_data)

print(f"✓ Loaded {len(vector)} vectors with {vector.shape[1]} dimensions each.")

# Save the scaled vectors to a CSV file
vector_df = pd.DataFrame(vector, columns=features)
vector_df.to_csv('outputs/vectorized.csv', index=False)
print(f"✓ Saved scaled vectors to outputs/vectorized.csv")