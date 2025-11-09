import pandas as pd

ENCODED_INPUT = "outputs/encoded.csv"
TRANSACTION_OUTPUT = "outputs/transaction.csv"

# Load the encoded CSV
encoded_df = pd.read_csv(ENCODED_INPUT)

print(f"\n✓ Loaded {len(encoded_df)} rows from {ENCODED_INPUT}")
print(f"✓ Columns: {list(encoded_df.columns)}")
print("\nNote: ClaimID_hash, PatientID_hash, and ProviderID_hash will be kept as single columns with original values")

# Define discretization functions
def discretize_claim_amount(amount):
    """Discretize claim amount into bins"""
    amount = float(amount)
    if amount < 1000:
        return 'low'
    elif 1000 <= amount < 3000:
        return 'medium'
    elif 3000 <= amount < 7000:
        return 'high'
    else:
        return 'very_high'

def discretize_age(age):
    """Discretize patient age into bins"""
    age = float(age)
    if age < 18:
        return 'minor'
    elif 18 <= age < 30:
        return 'young_adult'
    elif 30 <= age < 50:
        return 'middle_age'
    elif 50 <= age < 70:
        return 'senior'
    else:
        return 'elderly'

def discretize_income(income):
    """Discretize patient income into bins"""
    income = float(income)
    if income < 40000:
        return 'low_income'
    elif 40000 <= income < 70000:
        return 'medium_income'
    elif 70000 <= income < 100000:
        return 'high_income'
    else:
        return 'very_high_income'

def discretize_date(timestamp):
    """Discretize claim date by year"""
    try:
        dt = datetime.fromtimestamp(int(timestamp))
        year = dt.year
        if year < 2022:
            return 'before_2022'
        elif year == 2022:
            return 'year_2022'
        elif year == 2023:
            return 'year_2023'
        else:
            return 'year_2024_plus'
    except:
        return 'unknown_date'

def discretize_hash_id(hash_id, prefix, num_bins=10):
    """Discretize hash IDs into bins"""
    hash_id = int(hash_id)
    bin_num = hash_id % num_bins
    return f'{prefix}_bin_{bin_num}'

def get_hash_value(hash_id):
    """Get original hash ID value (no discretization)"""
    return int(hash_id)

def discretize_target_encoding(value, prefix, thresholds=None):
    """Discretize target encoding values"""
    value = float(value)
    if thresholds is None:
        # Use quartiles as default
        if value < 2000:
            return f'{prefix}_low'
        elif 2000 <= value < 5000:
            return f'{prefix}_medium'
        elif 5000 <= value < 8000:
            return f'{prefix}_high'
        else:
            return f'{prefix}_very_high'
    else:
        # Custom thresholds
        if value < thresholds[0]:
            return f'{prefix}_low'
        elif value < thresholds[1]:
            return f'{prefix}_medium'
        elif value < thresholds[2]:
            return f'{prefix}_high'
        else:
            return f'{prefix}_very_high'

# Define category values for one-hot encoding
# ClaimID_hash, PatientID_hash, and ProviderID_hash will be kept as single columns with original values
CATEGORIES = {
    'ClaimAmount': ['low', 'medium', 'high', 'very_high'],
    'PatientAge': ['minor', 'young_adult', 'middle_age', 'senior', 'elderly'],
    'PatientIncome': ['low_income', 'medium_income', 'high_income', 'very_high_income'],
    'ClaimDate': ['before_2022', 'year_2022', 'year_2023', 'year_2024_plus', 'unknown_date'],
    'DiagnosisCode_target': ['DiagnosisCode_low', 'DiagnosisCode_medium', 'DiagnosisCode_high', 'DiagnosisCode_very_high'],
    'ProcedureCode_target': ['ProcedureCode_low', 'ProcedureCode_medium', 'ProcedureCode_high', 'ProcedureCode_very_high']
}
# Note: ClaimID_hash, PatientID_hash, and ProviderID_hash are NOT in CATEGORIES - they will be separate columns with original values

# Generate all column names for one-hot encoding
def generate_onehot_columns():
    """Generate all one-hot encoded column names"""
    columns = ['transaction_id']
    for feature, categories in CATEGORIES.items():
        for category in categories:
            # Target encodings already have prefix in category name
            # Other features need feature prefix added
            if feature.endswith('_target'):
                columns.append(category)  # Already has prefix (e.g., 'DiagnosisCode_low')
            else:
                columns.append(f'{feature}_{category}')  # Add prefix (e.g., 'ClaimAmount_low')
    # Add hash ID columns as separate columns (not one-hot encoded) - original values
    columns.append('ClaimID_hash')
    columns.append('PatientID_hash')
    columns.append('ProviderID_hash')
    return columns

print("✓ Discretization functions ready")
print("Note: ClaimID_hash, PatientID_hash, and ProviderID_hash will be kept as single columns with original values")

# Verify required functions are defined
try:
    get_hash_value(0)  # Test function exists
except NameError:
    raise NameError("get_hash_value function not found. Please run the 'Define discretization functions' cell first!")

# Process each row and create one-hot encoded binary transactions
transactions_data = []
transaction_id = 1

for idx, row in encoded_df.iterrows():
    # Discretize each column
    discretized_values = {
        'ClaimAmount': discretize_claim_amount(row['ClaimAmount']),
        'PatientAge': discretize_age(row['PatientAge']),
        'PatientIncome': discretize_income(row['PatientIncome']),
        'ClaimDate': discretize_date(row['ClaimDate']),
        'DiagnosisCode_target': discretize_target_encoding(row['DiagnosisCode_target'], 'DiagnosisCode'),
        'ProcedureCode_target': discretize_target_encoding(row['ProcedureCode_target'], 'ProcedureCode')
    }

    # Store hash ID original values separately (not one-hot encoded)
    claim_id_value = get_hash_value(row['ClaimID_hash'])
    patient_id_value = get_hash_value(row['PatientID_hash'])
    provider_id_value = get_hash_value(row['ProviderID_hash'])

    # Create one-hot encoded row (binary: 1 if matches, 0 otherwise)
    onehot_row = [transaction_id]

    # For each feature in CATEGORIES, create binary columns
    for feature, categories in CATEGORIES.items():
        value = discretized_values[feature]
        for category in categories:
            # Set to 1 if this category matches the discretized value, 0 otherwise
            onehot_row.append(1 if value == category else 0)

    # Add hash ID columns as regular columns (not one-hot encoded) - original values
    onehot_row.append(claim_id_value)
    onehot_row.append(patient_id_value)
    onehot_row.append(provider_id_value)

    transactions_data.append(onehot_row)
    transaction_id += 1

    # Generate column names
column_names = generate_onehot_columns()

# Create DataFrame with one-hot encoded binary values
transactions_df = pd.DataFrame(transactions_data, columns=column_names)

print(f"✓ Processed {len(transactions_df)} transactions")
print(f"✓ Total columns: {len(column_names)}")
print(f"  - One-hot binary columns: {len([c for c in column_names if c not in ['transaction_id', 'ClaimID_hash', 'PatientID_hash', 'ProviderID_hash']])}")
print(f"  - Hash ID columns (original values): 3")
print(f"\nSample columns (first 10): {column_names[:10]}")
print(f"\nSample transactions (first 3 rows, first 10 columns):")
print(transactions_df.iloc[:3, :10])

# Check binary values (excluding hash ID columns which have integer values)
binary_cols = [col for col in transactions_df.columns if col not in ['transaction_id', 'ClaimID_hash', 'PatientID_hash', 'ProviderID_hash']]
print(f"\n✓ All one-hot values are binary (0 or 1): {transactions_df[binary_cols].isin([0, 1]).all().all()}")
print(f"\nHash ID columns (original integer values):")
print(f"  ClaimID_hash: {transactions_df['ClaimID_hash'].min()} to {transactions_df['ClaimID_hash'].max()}")
print(f"  PatientID_hash: {transactions_df['PatientID_hash'].min()} to {transactions_df['PatientID_hash'].max()}")
print(f"  ProviderID_hash: {transactions_df['ProviderID_hash'].min()} to {transactions_df['ProviderID_hash'].max()}")

transactions_df.to_csv(TRANSACTION_OUTPUT, index=False)

print("Done!")
print(f"✓ Saved transactions to: {TRANSACTION_OUTPUT}")
print(f"✓ Total transactions: {len(transactions_df)}")
print(f"✓ Total columns: {len(transactions_df.columns)}")
print(f"  - One-hot binary columns: {len([c for c in transactions_df.columns if c not in ['transaction_id', 'ClaimID_hash', 'PatientID_hash', 'ProviderID_hash']])}")
print(f"  - Hash ID columns (original values): 3 (ClaimID_hash, PatientID_hash, ProviderID_hash)")
print(f"\nNote: One-hot values are binary (0/1), hash IDs retain original integer values. Ready for Apriori!")