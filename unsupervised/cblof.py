import pandas as pd

RULE_INPUT = "outputs/rules.csv"
CBLOF_OUTPUT = "outputs/cblof_scores.csv"

# Load the rules CSV and quick clean just in case
rules_df = pd.read_csv(RULE_INPUT)

print(rules_df.head())
print(rules_df.info())

rules_df.dropna(how='all', inplace=True)

if rules_df.empty:
    print("The Rules Data Frame is empty!")

# Identifying numeric columns of the data and droping missing values in numeric columns
numeric_columns = rules_df.select_dtypes(include="number").columns
print(numeric_columns)

rules_df.isnull().sum()
rules_df.dropna(subset=numeric_columns, inplace=True)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

