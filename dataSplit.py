import os, re
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


CLASSES = {
    "01. Normal": 0,
    "02. Variation from normal": 1,
    "03. OPMD": 2,
    "04. Oral Cancer": 3,
}

def extract_patient_id(filename: str) -> str:
    stem = os.path.splitext(filename)[0].strip()
    if m := re.match(r"^SMITA0*(\d+)", stem, re.IGNORECASE):
        return f"SMITA_{int(m.group(1))}"
    if m := re.match(r"^(\d+)\s*-", stem):
        return f"NUM_{int(m.group(1))}"
    if m := re.match(r"^Ca\s*(\d+)", stem, re.IGNORECASE):
        return f"CA_{int(m.group(1))}"
    return f"OTHER_{stem}"

# 1. Scan 01. Unannotated and deduplicate by keeping the highest-severity label
records = {}
for cls_name, label in CLASSES.items():
    root = os.path.join("SMART-OM", cls_name, "01. Unannotated")
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.startswith("."):
                continue
            rel_path = os.path.relpath(os.path.join(dirpath, f), "SMART-OM")
            if f not in records or label > records[f]["label"]:
                records[f] = {
                    "filepath": rel_path,
                    "filename": f,
                    "patient_id": extract_patient_id(f),
                    "class_name": cls_name,
                    "label": label,
                }
df = pd.DataFrame(records.values())


# 2. Patient-level stratified 80/20 split (take fold 0 as 20% test)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(df, df["label"], groups=df["patient_id"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)


# 3. Verify zero patient leakage
assert set(train_df["patient_id"]).isdisjoint(set(test_df["patient_id"]))
os.makedirs("splits", exist_ok=True)
train_df.to_csv("splits/train.csv", index=False)
test_df.to_csv("splits/test.csv", index=False)
test_df["filename"].to_csv("splits/exam_list.txt", index=False, header=False)

print("Shape : ")
print(train_df.shape)
print("##############################################")
print("Columns : ")
print(train_df.columns)
print("##############################################")
print("Count Null Val : ")
print(train_df.isna().sum())

print("##############################################")


total = len(train_df) + len(test_df)

pct_train = (len(train_df) / total ) * 100
pct_test = (len(test_df) / total ) * 100

print(f"Train : {pct_train:.2f}% ({len(train_df)} samples)")
print(f"Test  : {pct_test:.2f}% ({len(test_df)} samples)")
