import pandas as pd
import numpy as np
# TODO: Enable this in Google Colab
# from google.colab import drive
# drive.mount('/content/drive')
# RAW_DATA = "/content/drive/Shareddrives/Data Mining - Group Project/Group_Project/enhanced_health_insurance_claims.csv"

RAW_DATA = 'sample/enhanced_health_insurance_claims.csv'
ENCODED_OUTPUT = "claims_encoded_output.csv"
SAMPLE_OUTPUT = "samples/sample_output.csv"

EXPECTED_COLUMNS = [
    "ClaimID", "PatientID", "ProviderID", "ClaimAmount", "ClaimDate", "DiagnosisCode", "ProcedureCode", "PatientAge", "PatientGender", "ClaimStatus", "PatientIncome"
]

LOWCARD_COLUMNS = ["PatientGender", "ClaimStatus"]
HIGHCARD_COLUMNS = ["DiagnosisCode", "ProcedureCode"]
NUMERIC_COLUMNS = ["ClaimAmount", "PatientAge", "PatientIncome"]

#loading csv data
data_input = pd.read_csv(RAW_DATA);

#checking for missing data columns
missing_columns = []
for col in EXPECTED_COLUMNS:
    if col not in data_input.columns:
        missing_columns.append(col)

if len(missing_columns) > 0:
    print("Columns missing from your dataset: ")
    print(missing_columns)
    print("")
    raise SystemExit("Please find another dataset or fix code.")

data_input = data_input[EXPECTED_COLUMNS].copy()

#~~~~~~~~~~~~~~~~~~~~ Basic cleaning ~~~~~~~~~~~~~~~~~~~~
#cleaning blank/gap data, "error='coerce'" is to catch any data that cannot be converted into numeric values
for num_data in NUMERIC_COLUMNS:
    if num_data in data_input.columns:
        data_input[num_data] = pd.to_numeric(data_input[num_data], errors="coerce")

data_input["ClaimDate"] = pd.to_datetime(data_input["ClaimDate"], errors="coerce")

for missing_val in NUMERIC_COLUMNS:
    if missing_val in data_input.columns:
        med_val = data_input[missing_val].median()
        data_input[missing_val] = data_input[missing_val].fillna(med_val)

for replace_val in [
    "DiagnosisCode","ProcedureCode","ProviderSpecialty",
    "PatientGender","ClaimStatus","ClaimID","PatientID","ProviderID"
]:
    if replace_val in data_input.columns:
        mode_series = data_input[replace_val].mode(dropna=True)
        mode_val = mode_series.iloc[0] if len(mode_series) else "UNKNOWN"
        data_input[replace_val] = data_input[replace_val].fillna(mode_val)

if data_input["ClaimDate"].isna().any():
    if data_input["ClaimDate"].notna().any():
        mid = data_input["ClaimDate"].dropna().median()
    else:
        mid = pd.Timestamp("2024-01-01")
    data_input["ClaimDate"] = data_input["ClaimDate"].fillna(mid)

#filtering out abnormal number data
data_input["PatientAge"] = data_input["PatientAge"].clip(lower=0, upper=120)
data_input["ClaimAmount"] = data_input["ClaimAmount"].clip(lower=0)
data_input["PatientIncome"] = data_input["PatientIncome"].clip(lower=0)

data_input = data_input.drop_duplicates().reset_index(drop=True)

for id_col in ["ClaimID","PatientID","ProviderID"]:
    data_input[id_col + "_hash"] = pd.factorize(data_input[id_col].astype(str))[0].astype("int32")

data_input["ClaimDate"] = (data_input["ClaimDate"].astype("int64") // 10**9).astype("int64")

df_onehot = pd.get_dummies(data_input[LOWCARD_COLUMNS], prefix=LOWCARD_COLUMNS, dtype="int8")
df = pd.concat([data_input.drop(columns=LOWCARD_COLUMNS), df_onehot], axis=1)

global_mean = data_input["ClaimAmount"].mean()
for h_col in HIGHCARD_COLUMNS:
    means = data_input.groupby(h_col)["ClaimAmount"].mean()
    data_input[h_col + "_target"] = data_input[h_col].map(means).fillna(global_mean)

#~~~~~~~~~~~~~~~~~~~~ Selecting columns ~~~~~~~~~~~~~~~~~~~~
keep_columns = []
keep_columns += ["ClaimAmount", "PatientAge", "PatientIncome"]
keep_columns += ["ClaimDate", "ClaimID_hash", "PatientID_hash", "ProviderID_hash"]
keep_columns += [c + "_target" for c in HIGHCARD_COLUMNS]

keep_columns += [c for c in data_input.columns if c.startswith("PatientGender_") or c.startswith("ClaimStatus_")]
present_cols = [c for c in keep_columns if c in data_input.columns]
missing_cols = [c for c in keep_columns if c not in data_input.columns]
if missing_cols:
    print("Warning: missing columns skipped:", missing_cols)

encoded = data_input[keep_columns].copy()

#~~~~~~~~~~~~~~~~~~~~ Creating Encoded CSV ~~~~~~~~~~~~~~~~~~~~
import os
ENCODED_OUTPUT = "outputs/encoded.csv"
SAMPLE_OUTPUT = "outputs/sample.csv"
os.makedirs("outputs", exist_ok=True)
os.makedirs(os.path.dirname(ENCODED_OUTPUT), exist_ok=True)
os.makedirs(os.path.dirname(SAMPLE_OUTPUT), exist_ok=True)

encoded.to_csv(ENCODED_OUTPUT, index=False)

sample_n = 20 if len(encoded) >= 20 else len(encoded)
encoded.sample(n=sample_n, random_state=42).to_csv(SAMPLE_OUTPUT, index=False)

print("Done!")
print(f"Saved encoded dataset to: {ENCODED_OUTPUT}  (rows={len(encoded)}, cols={encoded.shape[1]})")
print(f"Saved mock sample to:      {SAMPLE_OUTPUT}")
