import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load transactions data frame
transactions_df = pd.read_csv('data/transactions.csv')

print(transactions_df.all)