# clean landmarks raw file and save cleaned version for knn

import os
import pandas as pd
import numpy as np

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

# Step 6 (Advanced): normalise landmarks, align mirrored samples, then remove intra-class outliers
# idea:
# 1) normalise each sample (wrist-centre + scale) so we compare shape not position/size
# 2) for each class, compute a mean "template" shape
# 3) for each sample, compare original vs mirrored and keep whichever matches the template better
# 4) remove only extreme outliers per class using a robust cutoff (Median + MAD) instead of forcing top 5%

MAD_K = 3.5  # higher = less aggressive, lower = more aggressive
MAD_EPS = 1e-12  # prevents edge-case issues if MAD is ~0
do_outlier_removal = True  # set False if you only want alignment and no removals

def normalise_row_to_vec(row: pd.Series) -> np.ndarray:
    # reshape f0..f62 into (21,3)
    pts = row[feature_cols].to_numpy(dtype=float).reshape(21, 3)

    # translate: move wrist (landmark 0) to origin so position doesn't matter
    wrist = pts[0].copy()
    pts = pts - wrist

    # scale: divide by max distance from wrist so size/zoom doesn't matter
    dists = np.sqrt((pts ** 2).sum(axis=1))
    scale = float(dists.max())
    if scale > 1e-9:
        pts = pts / scale

    # flatten back to 63
    return pts.reshape(-1)

# indices of x components in flattened vector (x,y,z repeated)
x_idx = np.arange(0, 63, 3)

def mirror_vec(v: np.ndarray) -> np.ndarray:
    # mirror across vertical axis by negating x AFTER wrist-centering
    v2 = v.copy()
    v2[x_idx] *= -1
    return v2

def flip_original_features_inplace(frame: pd.DataFrame, row_indices: np.ndarray) -> None:
    """
    If a sample was mirror-aligned in the normalised space (x *= -1),
    we want the saved features (the ones we train on) to match that choice.

    In original MediaPipe coordinates where x is in [0,1], a horizontal mirror is:
      x -> 1 - x
    y and z stay the same.

    This function applies that flip in-place to the frame's feature columns,
    only for the specified row indices.
    """
    if len(row_indices) == 0:
        return
    # x columns are f0,f3,f6,...
    frame.loc[frame.index[row_indices], x_cols] = 1.0 - frame.loc[frame.index[row_indices], x_cols]

before_rows = len(df)

# build normalised vectors for all samples (N,63)
norm_vectors = [normalise_row_to_vec(df.iloc[i]) for i in range(len(df))]
norm_vectors = np.vstack(norm_vectors)

labels = df["label"].to_numpy()

# first pass: compute initial means (templates) per class
class_means = {}
for lab in sorted(pd.Series(labels).unique()):
    idx = np.where(labels == lab)[0]
    if len(idx) > 0:
        class_means[lab] = norm_vectors[idx].mean(axis=0)

# second pass: align each sample by choosing original or mirrored, whichever is closer to its class mean
aligned_vectors = norm_vectors.copy()
mirror_chosen = np.zeros(len(df), dtype=bool)
mirrored_count = 0

for i in range(len(df)):
    lab = labels[i]
    mean_vec = class_means[lab]

    v = norm_vectors[i]
    v_m = mirror_vec(v)

    d_orig = np.linalg.norm(v - mean_vec)
    d_mirr = np.linalg.norm(v_m - mean_vec)

    if d_mirr < d_orig:
        aligned_vectors[i] = v_m
        mirror_chosen[i] = True
        mirrored_count += 1
    else:
        aligned_vectors[i] = v
        mirror_chosen[i] = False

log_lines.append(f"Mirror-aligned samples (chosen mirrored orientation): {mirrored_count} / {len(df)}")

# IMPORTANT: apply the mirror choice to the ORIGINAL feature columns so training uses the mirrored features
# (x -> 1 - x in original coordinate space)
mirrored_indices = np.where(mirror_chosen)[0]
flip_original_features_inplace(df, mirrored_indices)

# recompute class means using aligned vectors (better templates after alignment)
class_means_aligned = {}
for lab in sorted(pd.Series(labels).unique()):
    idx = np.where(labels == lab)[0]
    if len(idx) > 0:
        class_means_aligned[lab] = aligned_vectors[idx].mean(axis=0)

# compute distance-to-mean for aligned vectors (this is what we use for outlier detection)
dist_to_mean = np.zeros(len(df), dtype=float)
for lab in sorted(pd.Series(labels).unique()):
    idx = np.where(labels == lab)[0]
    if len(idx) < 5:
        continue
    mean_vec = class_means_aligned[lab]
    diffs = aligned_vectors[idx] - mean_vec
    dist_to_mean[idx] = np.sqrt((diffs ** 2).sum(axis=1))

# store distances for logging/debug
df["dist_to_class_mean"] = dist_to_mean
df["mirror_aligned"] = mirror_chosen

# optional: remove the most extreme outliers within each class (ROBUST: Median + MAD)
if do_outlier_removal:
    keep_mask = np.ones(len(df), dtype=bool)
    removed_by_label = {}
    cutoffs_by_label = {}

    for lab in sorted(pd.Series(labels).unique()):
        class_mask = (df["label"] == lab).to_numpy()
        class_dists = df.loc[class_mask, "dist_to_class_mean"].to_numpy()

        if len(class_dists) < 5:
            removed_by_label[lab] = 0
            cutoffs_by_label[lab] = np.nan
            continue

        med = float(np.median(class_dists))
        mad = float(np.median(np.abs(class_dists - med)))
        mad_safe = mad if mad > MAD_EPS else MAD_EPS

        cutoff = med + (MAD_K * mad_safe)
        cutoffs_by_label[lab] = cutoff

        class_keep = class_dists <= cutoff
        class_idx = np.where(class_mask)[0]
        keep_mask[class_idx] = class_keep

        removed_by_label[lab] = int((~class_keep).sum())

    df = df[keep_mask].copy()
    log_lines.append(f"Dropped intra-class outliers (Median + MAD, MAD_K={MAD_K}): {before_rows - len(df)}")
    log_lines.append("Outliers removed per class:")
    log_lines.append(pd.Series(removed_by_label).sort_index().to_string())
    log_lines.append("Cutoff per class (distance threshold):")
    log_lines.append(pd.Series(cutoffs_by_label).sort_index().to_string())

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
