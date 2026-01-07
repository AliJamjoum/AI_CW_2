#clean landmarks raw file and save cleaned version for knn

import os
import pandas as pd

# input and output paths
RAW_PATH = "../outputs/landmarks_raw.csv"
CLEAN_PATH = "../outputs/landmarks_clean.csv"
SUMMARY_PATH = "../outputs/preprocess_summary.txt"

# make sure outputs folder exists (prevents file save errors)
os.makedirs("../outputs", exist_ok=True)

#load raw landmarks table
df = pd.read_csv(RAW_PATH)

#identify feature columns (f0-f62)
feature_cols = [c for c in df.columns if c.startswith("f")]

# build x and y column lists (x is f0,f3,f6... and y is f1,f4,f7...), z woiuld be f2
x_cols = [f"f{i}" for i in range(0, 63, 3)]
y_cols = [f"f{i}" for i in range(1, 63, 3)]

#selects the label column. counts how mant times each label appears, sorts count by label name
def label_counts(frame: pd.DataFrame) -> pd.Series:
    return frame["label"].value_counts().sort_index()

#stores stats as we go
log_lines = []

#base snapshot - df.shape returns a tuple, f makes it formatted string
log_lines.append(f"Baseline shape: {df.shape}")
log_lines.append("Class counts (before):")
#counts how many samples and converts formatted series into text
log_lines.append(label_counts(df).to_string())

# STEP 1 KEEP EXPECTED LABELS

#set showing allowed asl labels
valid_labels = set(list("ABCDEFGHIJ"))
#no of rows before diltering
before_rows = len(df)

#boolean mask inside [] means a list/series of true/false values
#keeps only rows where mask is true, copies into a df
df = df[df["label"].isin(valid_labels)].copy()
#says how many are removed
log_lines.append(f"Removed invalid-label rows: {before_rows - len(df)}")

# Step 2: Ensure all feature columns are numeric (anything weird becomes NaN)
#selects only feature cols + runs a function to convert value to numbers. coerce = NaN
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")


# Step 3: Drop any rows with missing feature values
before_rows = len(df)
#removes rows with missing values in featues
df = df.dropna(subset=feature_cols).copy()
log_lines.append(f"Dropped rows with missing features: {before_rows - len(df)}")

# Step 4: Drop duplicate instance_id rows (keep first occurrence)
before_rows = len(df)
df = df.drop_duplicates(subset=["instance_id"], keep="first").copy()
log_lines.append(f"Dropped duplicate instance_id rows: {before_rows - len(df)}")

# Step 5: Drop rows with x or y outside [0, 1] (MediaPipe x,y are normalised)
before_rows = len(df)
x_ok = df[x_cols].ge(0).all(axis=1) & df[x_cols].le(1).all(axis=1)
y_ok = df[y_cols].ge(0).all(axis=1) & df[y_cols].le(1).all(axis=1)
df = df[x_ok & y_ok].copy()
log_lines.append(f"Dropped rows with x/y out of range: {before_rows - len(df)}")

# Final snapshot
log_lines.append(f"Final shape: {df.shape}")
log_lines.append("Class counts (after):")
log_lines.append(label_counts(df).to_string())

# Save cleaned dataset
df.to_csv(CLEAN_PATH, index=False)

# Save a preprocessing summary for your report and slides
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(log_lines) + "\n")

# Console output so you can quickly verify it worked
print("Saved cleaned data to:", CLEAN_PATH)
print("Saved summary to:", SUMMARY_PATH)
print("\n".join(log_lines))