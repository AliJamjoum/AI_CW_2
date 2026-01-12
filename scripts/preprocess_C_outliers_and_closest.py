# preprocess_C_outliers_and_closest.py
# Advanced: For label C only:
# - normalise landmarks (wrist-centre + scale)
# - mirror-align each sample (choose original vs mirrored based on closeness to class mean)
# - compute distance to class mean
# - EXCLUDE outliers using a ROBUST, EVIDENCE-BASED cutoff (Median + MAD) instead of fixed 5%
# - print and save the filenames of excluded samples
# - list which samples were better mirrored (mirror_aligned == True), save to CSV/TXT
# - find the single sample closest to mean and copy its image to outputs/

import os
import shutil
import numpy as np
import pandas as pd

RAW_PATH = "../outputs/landmarks_raw.csv"
OUT_DIR = "../outputs"
LABEL = "C"

# where the images live for this label
DATA_LABEL_DIR = f"../data/{LABEL}"

# outputs
CLEAN_C_PATH = "../outputs/landmarks_clean_C_only.csv"
EXCLUDED_CSV = "../outputs/excluded_C_samples.csv"
EXCLUDED_TXT = "../outputs/excluded_C_samples.txt"
CLOSEST_TXT = "../outputs/closest_to_mean_C.txt"
CLOSEST_IMAGE_OUT = "../outputs/closest_to_mean_C_image"

# mirrored list outputs
MIRRORED_ALL_CSV = "../outputs/mirrored_C_samples_all.csv"
MIRRORED_KEPT_CSV = "../outputs/mirrored_C_samples_kept.csv"
MIRRORED_EXCLUDED_CSV = "../outputs/mirrored_C_samples_excluded.csv"
MIRRORED_ALL_TXT = "../outputs/mirrored_C_samples_all.txt"
MIRRORED_KEPT_TXT = "../outputs/mirrored_C_samples_kept.txt"
MIRRORED_EXCLUDED_TXT = "../outputs/mirrored_C_samples_excluded.txt"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- robust outlier settings (NEW) ----------
# A common robust rule: keep distances <= median + (k * MAD)
# k=3.5 is widely used as a conservative threshold.
MAD_K = 3.5

# If MAD is ~0 (rare but possible if data is extremely tight), we fall back to a tiny epsilon.
MAD_EPS = 1e-12

# ---------- helpers ----------

def pick_id_column(frame: pd.DataFrame) -> str:
    for col in ["file", "filename", "image", "img", "path", "img_path", "image_path"]:
        if col in frame.columns:
            return col
    if "instance_id" in frame.columns:
        return "instance_id"
    return ""

def normalise_row_to_vec(row: pd.Series, feature_cols: list[str]) -> np.ndarray:
    pts = row[feature_cols].to_numpy(dtype=float).reshape(21, 3)

    wrist = pts[0].copy()
    pts = pts - wrist

    dists = np.sqrt((pts ** 2).sum(axis=1))
    scale = float(dists.max())
    if scale > 1e-9:
        pts = pts / scale

    return pts.reshape(-1)

def mirror_vec(v: np.ndarray) -> np.ndarray:
    x_idx = np.arange(0, 63, 3)
    v2 = v.copy()
    v2[x_idx] *= -1
    return v2

def reconstruct_flipped_original_coords(row: pd.Series, feature_cols: list[str]) -> np.ndarray:
    v = row[feature_cols].to_numpy(dtype=float).reshape(21, 3)
    v[:, 0] = 1.0 - v[:, 0]
    return v.reshape(-1)

def guess_image_path(id_value: str) -> str | None:
    """
    Handles:
    - full/relative paths that already exist
    - filenames like "C_sample_441.jpg"
    - identifiers like "C/C_sample_441.jpg" (common in your output)
    """
    if not isinstance(id_value, str) or not id_value:
        return None

    if os.path.exists(id_value):
        return id_value

    norm = id_value.replace("\\", "/")

    # strip leading "C/" if present
    if norm.startswith(f"{LABEL}/"):
        norm = norm[len(f"{LABEL}/"):]

    candidate = os.path.join(DATA_LABEL_DIR, norm)
    if os.path.exists(candidate):
        return candidate

    return None

def identifiers_from(frame: pd.DataFrame, id_col: str) -> list[str]:
    if len(frame) == 0:
        return []
    if id_col:
        return frame[id_col].astype(str).tolist()
    return [str(i) for i in frame.index.tolist()]

def write_list_txt(path: str, header_lines: list[str], items: list[str]) -> None:
    with open(path, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write("\n")
        for it in items:
            f.write(it + "\n")

# ---------- load + basic cleaning (C only) ----------

df = pd.read_csv(RAW_PATH)

feature_cols = [c for c in df.columns if c.startswith("f")]
if len(feature_cols) != 63:
    raise ValueError(f"Expected 63 feature columns f0..f62, found {len(feature_cols)}")

if "label" not in df.columns:
    raise ValueError("landmarks_raw.csv must contain a 'label' column")

df = df[df["label"] == LABEL].copy()

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=feature_cols).copy()

id_col = pick_id_column(df)

# ---------- build normalised vectors ----------

norm_vectors = [normalise_row_to_vec(df.iloc[i], feature_cols) for i in range(len(df))]
norm_vectors = np.vstack(norm_vectors)

# initial mean in normalised space
mean_vec = norm_vectors.mean(axis=0)

# ---------- mirror-align each sample against mean ----------

aligned_is_mirrored = np.zeros(len(df), dtype=bool)
aligned_vectors = norm_vectors.copy()

for i in range(len(df)):
    v = norm_vectors[i]
    v_m = mirror_vec(v)

    d_orig = np.linalg.norm(v - mean_vec)
    d_mirr = np.linalg.norm(v_m - mean_vec)

    if d_mirr < d_orig:
        aligned_vectors[i] = v_m
        aligned_is_mirrored[i] = True

# recompute mean after alignment
mean_vec_aligned = aligned_vectors.mean(axis=0)

diffs = aligned_vectors - mean_vec_aligned
dist_to_mean = np.sqrt((diffs ** 2).sum(axis=1))

df["dist_to_class_mean"] = dist_to_mean
df["mirror_aligned"] = aligned_is_mirrored

# ---------- robust outlier cutoff using Median + MAD (REPLACES 5% RULE) ----------

median_dist = float(np.median(dist_to_mean))
mad = float(np.median(np.abs(dist_to_mean - median_dist)))

# guard against near-zero MAD
mad_safe = mad if mad > MAD_EPS else MAD_EPS

cutoff = median_dist + (MAD_K * mad_safe)
keep_mask = dist_to_mean <= cutoff

excluded = df.loc[~keep_mask].copy()
kept = df.loc[keep_mask].copy()

# ---------- mirror-aligned lists + counts ----------

mirrored_all = df[df["mirror_aligned"] == True].copy()
mirrored_kept = kept[kept["mirror_aligned"] == True].copy()
mirrored_excluded = excluded[excluded["mirror_aligned"] == True].copy()

mirrored_all_names = identifiers_from(mirrored_all, id_col)
mirrored_kept_names = identifiers_from(mirrored_kept, id_col)
mirrored_excluded_names = identifiers_from(mirrored_excluded, id_col)

total_count = len(df)
kept_count = len(kept)
excluded_count = len(excluded)

mirrored_total_count = len(mirrored_all)
mirrored_kept_count = len(mirrored_kept)
mirrored_excluded_count = len(mirrored_excluded)

# ---------- find closest-to-mean sample (among kept) ----------

if kept_count == 0:
    raise RuntimeError("All samples were excluded by the cutoff. Lower MAD_K or inspect dist_to_class_mean distribution.")

kept_idx = np.where(keep_mask)[0]
closest_global_i = kept_idx[np.argmin(dist_to_mean[keep_mask])]
closest_row = df.iloc[closest_global_i]

# ---------- apply alignment to ORIGINAL features before saving ----------

aligned_feature_matrix = df[feature_cols].to_numpy(dtype=float).copy()

for i in range(len(df)):
    if aligned_is_mirrored[i]:
        aligned_feature_matrix[i, :] = reconstruct_flipped_original_coords(df.iloc[i], feature_cols)

kept_features = aligned_feature_matrix[keep_mask]
kept.loc[:, feature_cols] = kept_features

# ---------- save outputs ----------

kept.to_csv(CLEAN_C_PATH, index=False)
excluded.to_csv(EXCLUDED_CSV, index=False)

mirrored_all.to_csv(MIRRORED_ALL_CSV, index=False)
mirrored_kept.to_csv(MIRRORED_KEPT_CSV, index=False)
mirrored_excluded.to_csv(MIRRORED_EXCLUDED_CSV, index=False)

# excluded names list
excluded_names = identifiers_from(excluded, id_col)

excluded_header = [
    f"Excluded samples for label {LABEL}",
    "Cutoff method: Median + MAD",
    f"MAD_K: {MAD_K}",
    f"Median distance: {median_dist:.6f}",
    f"MAD: {mad:.6f}",
    f"Cutoff: {cutoff:.6f}",
    f"Total rows (after NaN drop): {total_count}",
    f"Kept: {kept_count}",
    f"Excluded: {excluded_count}",
    f"Mirror-aligned total: {mirrored_total_count}",
    f"Mirror-aligned kept: {mirrored_kept_count}",
    f"Mirror-aligned excluded: {mirrored_excluded_count}",
]
write_list_txt(EXCLUDED_TXT, excluded_header, excluded_names)

# mirrored lists as TXT
mirrored_all_header = [
    f"Mirror-aligned samples for label {LABEL} (ALL)",
    "Definition: samples where mirrored version was closer to class mean than original.",
    f"Count: {mirrored_total_count} / {total_count}",
]
write_list_txt(MIRRORED_ALL_TXT, mirrored_all_header, mirrored_all_names)

mirrored_kept_header = [
    f"Mirror-aligned samples for label {LABEL} (KEPT)",
    f"Count: {mirrored_kept_count} / {kept_count}",
]
write_list_txt(MIRRORED_KEPT_TXT, mirrored_kept_header, mirrored_kept_names)

mirrored_excluded_header = [
    f"Mirror-aligned samples for label {LABEL} (EXCLUDED)",
    f"Count: {mirrored_excluded_count} / {excluded_count}",
]
write_list_txt(MIRRORED_EXCLUDED_TXT, mirrored_excluded_header, mirrored_excluded_names)

# closest sample info + copy its image if possible
closest_id_val = str(closest_row[id_col]) if id_col else str(closest_row.name)
closest_info_lines = []
closest_info_lines.append(f"Label: {LABEL}")
closest_info_lines.append(f"Closest-to-mean identifier: {closest_id_val}")
closest_info_lines.append(f"Distance to mean: {float(closest_row['dist_to_class_mean']):.6f}")
closest_info_lines.append(f"Was mirror-aligned: {bool(closest_row['mirror_aligned'])}")
closest_info_lines.append("")

closest_info_lines.append("Cutoff summary (Median + MAD):")
closest_info_lines.append(f"MAD_K: {MAD_K}")
closest_info_lines.append(f"Median distance: {median_dist:.6f}")
closest_info_lines.append(f"MAD: {mad:.6f}")
closest_info_lines.append(f"Cutoff: {cutoff:.6f}")
closest_info_lines.append("")

closest_info_lines.append("Counts summary:")
closest_info_lines.append(f"Total rows (after NaN drop): {total_count}")
closest_info_lines.append(f"Kept: {kept_count}")
closest_info_lines.append(f"Excluded: {excluded_count}")
closest_info_lines.append(f"Mirror-aligned total: {mirrored_total_count}")
closest_info_lines.append(f"Mirror-aligned kept: {mirrored_kept_count}")
closest_info_lines.append(f"Mirror-aligned excluded: {mirrored_excluded_count}")

img_path = guess_image_path(closest_id_val) if id_col else None
if img_path is not None:
    ext = os.path.splitext(img_path)[1]
    out_img = CLOSEST_IMAGE_OUT + ext
    shutil.copyfile(img_path, out_img)
    closest_info_lines.append("")
    closest_info_lines.append(f"Copied closest image to: {out_img}")
else:
    closest_info_lines.append("")
    closest_info_lines.append("Could not locate closest image file automatically (no usable filename/path).")

with open(CLOSEST_TXT, "w") as f:
    f.write("\n".join(closest_info_lines) + "\n")

# ---------- console output ----------

print(f"Label {LABEL} rows (after NaN drop): {total_count}")

print("\nCutoff method: Median + MAD")
print(f" - MAD_K: {MAD_K}")
print(f" - Median distance: {median_dist:.6f}")
print(f" - MAD: {mad:.6f}")
print(f" - Cutoff: {cutoff:.6f}")

print(f"\nKept: {kept_count}")
print(f"Excluded: {excluded_count}")

print(f"\nMirror-aligned (better mirrored) counts:")
print(f" - Total mirrored: {mirrored_total_count}")
print(f" - Kept mirrored: {mirrored_kept_count}")
print(f" - Excluded mirrored: {mirrored_excluded_count}")

print(f"\nSaved kept (aligned) C-only dataset to: {CLEAN_C_PATH}")
print(f"Saved excluded CSV to: {EXCLUDED_CSV}")
print(f"Saved excluded names TXT to: {EXCLUDED_TXT}")

print(f"\nSaved mirrored lists:")
print(f" - All mirrored CSV: {MIRRORED_ALL_CSV}")
print(f" - Kept mirrored CSV: {MIRRORED_KEPT_CSV}")
print(f" - Excluded mirrored CSV: {MIRRORED_EXCLUDED_CSV}")
print(f" - All mirrored TXT: {MIRRORED_ALL_TXT}")
print(f" - Kept mirrored TXT: {MIRRORED_KEPT_TXT}")
print(f" - Excluded mirrored TXT: {MIRRORED_EXCLUDED_TXT}")

print(f"\nSaved closest-to-mean info to: {CLOSEST_TXT}")

print("\nExcluded identifiers (first 30):")
for name in excluded_names[:30]:
    print(" -", name)

print("\nMirror-aligned identifiers (first 30):")
for name in mirrored_all_names[:30]:
    print(" -", name)

print("\nClosest-to-mean identifier:")
print(" -", closest_id_val)
