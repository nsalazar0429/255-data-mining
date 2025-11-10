# 🧠 255 Data Mining Project - Fraud Detection
# Setup & Execution Guide (MAC OS)

This README provides the complete step-by-step instructions to set up your environment, install dependencies, and run the data preprocessing and Apriori analysis scripts successfully on macOS using **Homebrew Python** and a **virtual environment**.

---

## ⚙️ Full Setup & Execution Instructions

```bash
# 1️⃣ Ensure Homebrew and Python 3 are up to date
brew update
brew upgrade python

# Verify Python installation
which python3
python3 --version
# Expected output:
# /opt/homebrew/bin/python3
# Python 3.14.x

# 2️⃣ Navigate to your project directory
cd ~/Documents/255-data-mining

# 3️⃣ Create a virtual environment
/opt/homebrew/bin/python3 -m venv .venv

# 4️⃣ Activate the virtual environment
source .venv/bin/activate
# Your terminal prompt should now show (.venv)

# 5️⃣ Upgrade pip, setuptools, and wheel
python3 -m pip install --upgrade pip setuptools wheel

# 6️⃣ Install required Python packages
python3 -m pip install pandas numpy scikit-learn mlxtend

# ✅ Verify installations
python3 -m pip list
# Expected key packages:
# pandas, numpy, scikit-learn, mlxtend, matplotlib

# 7️⃣ Run preprocessing script one
python3 pre_processing_one.py
# Expected output:
# Done!
# Saved encoded dataset to: outputs/encoded.csv  (rows=4500, cols=9)
# Saved mock sample to:      outputs/sample.csv

# 8️⃣ Run preprocessing script two
# ⚠️ Note: filename is pre_procesing_two.py (single “s” in procesing)
python3 pre_procesing_two.py
# Expected output:
# ✓ Loaded 4500 rows from outputs/encoded.csv
# ✓ Columns: ['ClaimAmount', 'PatientAge', 'PatientIncome', 'ClaimDate', ...]
# ✓ Processed 4500 transactions
# ✓ Total columns: 30
# ✓ Saved transactions to: outputs/transaction.csv

# 9️⃣ Run Apriori association rule mining
python3 apriori.py
# Expected output:
# --- Top 5 Rules Found ---
# [ table of rules ]
# Successfully found rules and saved them to rules.csv!

# 🔍 Verify generated outputs
ls outputs/
# Expected files:
# encoded.csv
# sample.csv
# transaction.csv
# rules.csv

# 🔧 Optional: check output content
head outputs/encoded.csv
head outputs/transaction.csv
head outputs/rules.csv

# 🔚 Deactivate the virtual environment when finished
deactivate
```
