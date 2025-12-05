import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load transactions data frame
transactions_df = pd.read_csv('outputs/transaction.csv')

# 1. Identify the columns you want to exclude
columns_to_exclude = ['transaction_id','ClaimID_hash','PatientID_hash','ProviderID_hash', 'ClaimDate_unknown_date']

# 2. Create the final DataFrame by dropping the unwanted columns
#    The 'axis=1' ensures that we drop columns, not rows.
#    The 'errors="ignore"' prevents the code from crashing if a column doesn't exist.
transactions_df_for_apriori = transactions_df.drop(
    columns=columns_to_exclude, 
    axis=1, 
    errors='ignore'
)

# Find common values using the apriori algorithm
frequent_itemsets = apriori(transactions_df_for_apriori, min_support=0.01, use_colnames=True)

# Generate the association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

# Add a filter for lift (e.g., lift >= 1.2)
rules = rules[rules['lift'] >= 1.2]

#sort rules by confidence in descending order
rules_sorted = rules.sort_values(by='confidence', ascending=False)

# Round all numeric values to 2 decimal places
rules_sorted = rules_sorted.round(2)

# Format the 'antecedents' and 'consequents' columns
# This converts the 'frozenset' object to a clean, readable string
rules_sorted['antecedents'] = rules_sorted['antecedents'].apply(lambda x: ', '.join(list(x)))
rules_sorted['consequents'] = rules_sorted['consequents'].apply(lambda x: ', '.join(list(x)))

# 1. Identify the columns you want to exclude
columns_to_exclude_from_output = ['conviction']

# 2. Create the final DataFrame by dropping the unwanted columns
# 2. Create the final DataFrame by dropping the unwanted columns
#    The 'axis=1' ensures that we drop columns, not rows.
#    The 'errors="ignore"' prevents the code from crashing if a column doesn't exist.
rules_filtered = rules_sorted.drop(
    columns=columns_to_exclude_from_output, 
    axis=1, 
    errors='ignore'
)

# Display the top 5 rules
print("--- Top 5 Rules Found ---")
print(rules_filtered.head())
print("-------------------------")

# Save all the sorted rules to our final output file
rules_filtered.to_csv('outputs/rules.csv', index=False)

print("Successfully found rules and saved them to rules.csv!")