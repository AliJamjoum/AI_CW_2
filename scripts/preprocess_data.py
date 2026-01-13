

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


RAW_PATH = "../outputs/landmarks_raw.csv"
CLEAN_PATH = "../outputs/landmarks_clean.csv"
SUMMARY_PATH = "../outputs/preprocess_summary.txt"


VIS_DIR = "../outputs/preprocess_visuals"
os.makedirs(VIS_DIR, exist_ok=True)


MAKE_VISUALS = True
HIST_CLASS = "B"    
PCA_SEED = 2025
PCA_MAX_POINTS = 2500  


os.makedirs("../outputs", exist_ok=True)


df = pd.read_csv(RAW_PATH)


feature_cols = [c for c in df.columns if c.startswith("f")]


x_cols = [f"f{i}" for i in range(0, 63, 3)]
y_cols = [f"f{i}" for i in range(1, 63, 3)]

def label_counts(frame: pd.DataFrame) -> pd.Series:
    return frame["label"].value_counts().sort_index()


log_lines = []


log_lines.append(f"Baseline shape: {df.shape}")
log_lines.append("Class counts (before):")
log_lines.append(label_counts(df).to_string())

# STEP 1 KEEP EXPECTED LABELS

valid_labels = set(list("ABCDEFGHIJ"))
before_rows = len(df)
df = df[df["label"].isin(valid_labels)].copy()
log_lines.append(f"Removed invalid-label rows: {before_rows - len(df)}")

# Step 2: Ensure all feature columns are numeric (anything weird becomes NaN)

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

# Step 3: Drop any rows with missing feature values

before_rows = len(df)
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

# Step 6: normalise landmarks, align mirrored samples, then remove intra-class outliers

MAD_K = 3.5
MAD_EPS = 1e-12
do_outlier_removal = True

def normalise_row_to_vec(row: pd.Series) -> np.ndarray:
    pts = row[feature_cols].to_numpy(dtype=float).reshape(21, 3)

    
    wrist = pts[0].copy()
    pts = pts - wrist

    
    dists = np.sqrt((pts ** 2).sum(axis=1))
    scale = float(dists.max())
    if scale > 1e-9:
        pts = pts / scale

    return pts.reshape(-1)


x_idx = np.arange(0, 63, 3)

def mirror_vec(v: np.ndarray) -> np.ndarray:
    v2 = v.copy()
    v2[x_idx] *= -1
    return v2

def flip_original_features_inplace(frame: pd.DataFrame, row_indices: np.ndarray) -> None:
    """
    If a sample was mirror-aligned in the normalised space (x *= -1),
    flip the ORIGINAL MediaPipe x coordinates so training uses the mirrored features:
      x -> 1 - x
    """
    if len(row_indices) == 0:
        return
    frame.loc[frame.index[row_indices], x_cols] = 1.0 - frame.loc[frame.index[row_indices], x_cols]

before_rows = len(df)


norm_vectors = [normalise_row_to_vec(df.iloc[i]) for i in range(len(df))]
norm_vectors = np.vstack(norm_vectors)

labels = df["label"].to_numpy()


class_means = {}
for lab in sorted(pd.Series(labels).unique()):
    idx = np.where(labels == lab)[0]
    if len(idx) > 0:
        class_means[lab] = norm_vectors[idx].mean(axis=0)


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


mirrored_indices = np.where(mirror_chosen)[0]
flip_original_features_inplace(df, mirrored_indices)


class_means_aligned = {}
for lab in sorted(pd.Series(labels).unique()):
    idx = np.where(labels == lab)[0]
    if len(idx) > 0:
        class_means_aligned[lab] = aligned_vectors[idx].mean(axis=0)


dist_to_mean = np.zeros(len(df), dtype=float)
for lab in sorted(pd.Series(labels).unique()):
    idx = np.where(labels == lab)[0]
    if len(idx) < 5:
        continue
    mean_vec = class_means_aligned[lab]
    diffs = aligned_vectors[idx] - mean_vec
    dist_to_mean[idx] = np.sqrt((diffs ** 2).sum(axis=1))


df["dist_to_class_mean"] = dist_to_mean
df["mirror_aligned"] = mirror_chosen


df_before_outliers = df.copy()
aligned_vectors_before_outliers = aligned_vectors.copy()
labels_before_outliers = labels.copy()


removed_by_label = {lab: 0 for lab in sorted(pd.Series(labels).unique())}
cutoffs_by_label = {lab: np.nan for lab in sorted(pd.Series(labels).unique())}


if do_outlier_removal:
    keep_mask = np.ones(len(df), dtype=bool)

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

    keep_mask_before_filter = keep_mask.copy()

    df = df[keep_mask].copy()

    aligned_vectors_after_outliers = aligned_vectors_before_outliers[keep_mask_before_filter]
    labels_after_outliers = labels_before_outliers[keep_mask_before_filter]

    log_lines.append(f"Dropped intra-class outliers (Median + MAD, MAD_K={MAD_K}): {before_rows - len(df)}")
    log_lines.append("Outliers removed per class:")
    log_lines.append(pd.Series(removed_by_label).sort_index().to_string())
    log_lines.append("Cutoff per class (distance threshold):")
    log_lines.append(pd.Series(cutoffs_by_label).sort_index().to_string())
else:
    aligned_vectors_after_outliers = aligned_vectors_before_outliers
    labels_after_outliers = labels_before_outliers


log_lines.append(f"Final shape: {df.shape}")
log_lines.append("Class counts (after):")
log_lines.append(label_counts(df).to_string())

# -------------------------
# VISUALISERS
# -------------------------
def _maybe_downsample(X: np.ndarray, y: np.ndarray, max_points: int, seed: int):
    if len(X) <= max_points:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_points, replace=False)
    return X[idx], y[idx]

def save_distance_histogram_for_class(frame_before: pd.DataFrame, lab: str, cutoffs: dict, out_path: str):
    if "dist_to_class_mean" not in frame_before.columns:
        print("No dist_to_class_mean column found, skipping histogram.")
        return

    subset = frame_before[frame_before["label"] == lab]
    if len(subset) == 0:
        print(f"No rows for class {lab}, skipping histogram.")
        return

    d = subset["dist_to_class_mean"].to_numpy()
    cutoff = cutoffs.get(lab, np.nan)

    plt.figure(figsize=(7, 5))
    plt.hist(d, bins=30)
    plt.title(f"Distance to class mean for class {lab} (before outlier removal)")
    plt.xlabel("Distance to class mean")
    plt.ylabel("Count")

    if np.isfinite(cutoff):
        plt.axvline(cutoff, linewidth=2, label=f"MAD cutoff = {cutoff:.3f}")
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_pca_scatter(X: np.ndarray, y: np.ndarray, out_path: str, title: str, seed: int):
 
    X_ds, y_ds = _maybe_downsample(X, y, PCA_MAX_POINTS, seed)

    pca = PCA(n_components=2, random_state=seed)
    X2 = pca.fit_transform(X_ds)

    plt.figure(figsize=(7, 5))
    
    labs_sorted = sorted(pd.Series(y_ds).unique())
    lab_to_int = {lab: i for i, lab in enumerate(labs_sorted)}
    c = np.array([lab_to_int[v] for v in y_ds])

    plt.scatter(X2[:, 0], X2[:, 1], c=c, s=10)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

if MAKE_VISUALS:
    
    save_distance_histogram_for_class(
        frame_before=df_before_outliers,
        lab=HIST_CLASS,
        cutoffs=cutoffs_by_label,
        out_path=os.path.join(VIS_DIR, f"hist_dist_class_{HIST_CLASS}.png"),
    )

    
    save_pca_scatter(
        X=aligned_vectors_before_outliers,
        y=labels_before_outliers,
        out_path=os.path.join(VIS_DIR, "pca_before_outlier_removal.png"),
        title="PCA of aligned landmark vectors (before outlier removal)",
        seed=PCA_SEED,
    )

    
    save_pca_scatter(
        X=aligned_vectors_after_outliers,
        y=labels_after_outliers,
        out_path=os.path.join(VIS_DIR, "pca_after_outlier_removal.png"),
        title="PCA of aligned landmark vectors (after outlier removal)",
        seed=PCA_SEED,
    )

    
    before_counts = label_counts(df_before_outliers)
    after_counts = label_counts(df)

    classes_sorted = list(before_counts.index)
    retained = np.array([after_counts.get(c, 0) for c in classes_sorted])
    removed = np.array([before_counts.get(c, 0) for c in classes_sorted]) - retained

    plt.figure(figsize=(8, 5))
    plt.bar(classes_sorted, retained, label="Retained after preprocessing", color="#1f4fd8") 
    plt.bar(classes_sorted, removed, bottom=retained, label="Removed during preprocessing", color="#9ec9ff") 
    plt.title("Effect of preprocessing on class distributions")
    plt.xlabel("Class label")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "stacked_counts_retained_removed.png"), dpi=200)
    plt.close()


df.to_csv(CLEAN_PATH, index=False)


with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(log_lines) + "\n")


print("Saved cleaned data to:", CLEAN_PATH)
print("Saved summary to:", SUMMARY_PATH)
print("Saved visuals to:", VIS_DIR)
print("\n".join(log_lines))
