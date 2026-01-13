import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# -------------------------
# Configure
# -------------------------
LABEL = "C"
REQUIRE_FLIPPED = True          # choose a sample that was flipped by preprocessing
FORCE_INSTANCE_ID = None        # set to "C/C_sample_94.jpg" to force

SAVE_CLEAN_OVERLAY_ON_FLIPPED_IMAGE = True  # useful to demonstrate the correction visually

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

FEATURE_COLS = [f"f{i}" for i in range(63)]

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.normpath(os.path.join(HERE, "..", "outputs", "landmarks_raw.csv"))
CLEAN_CSV = os.path.normpath(os.path.join(HERE, "..", "outputs", "landmarks_clean.csv"))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "outputs", "preprocess_visuals"))
os.makedirs(OUT_DIR, exist_ok=True)


def row_to_points_xyz(row: pd.Series) -> np.ndarray:
    return row[FEATURE_COLS].astype(float).to_numpy().reshape(21, 3)


def compute_steps(points_xyz: np.ndarray):
    raw_xy = points_xyz[:, :2].astype(float)
    wrist = raw_xy[0].copy()
    translated = raw_xy - wrist
    d = np.sqrt((translated[:, 0] ** 2) + (translated[:, 1] ** 2))
    max_d = float(d.max()) if float(d.max()) != 0.0 else 1.0
    scaled = translated / max_d
    mirrored = scaled.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    return raw_xy, translated, scaled, mirrored


def normalise_hand_space(points_xyz: np.ndarray) -> np.ndarray:
    xy = points_xyz[:, :2].astype(float)
    centred = xy - xy[0]
    d = np.sqrt((centred[:, 0] ** 2) + (centred[:, 1] ** 2))
    max_d = float(d.max()) if float(d.max()) != 0.0 else 1.0
    scaled = centred / max_d
    return scaled.reshape(-1)


def pick_best_row(df_clean: pd.DataFrame, label: str, require_flipped: bool) -> pd.Series:
    sub = df_clean[df_clean["label"] == label].copy()
    if sub.empty:
        raise ValueError(f"No rows found for label '{label}' in landmarks_clean.csv")

    if require_flipped:
        if "mirror_flipped" not in sub.columns:
            raise ValueError("mirror_flipped column not found in landmarks_clean.csv")
        sub = sub[sub["mirror_flipped"] == True].copy()
        if sub.empty:
            raise ValueError(f"No rows with label '{label}' and mirror_flipped==True")

    vecs = np.stack([normalise_hand_space(row_to_points_xyz(r)) for _, r in sub.iterrows()])
    template = vecs.mean(axis=0)
    dists = np.linalg.norm(vecs - template, axis=1)
    return sub.iloc[int(np.argmin(dists))]


def plot_points(ax, pts: np.ndarray, title: str, xlim=None, ylim=None):
    ax.scatter(pts[:, 0], pts[:, 1], s=28)
    for a, b in HAND_CONNECTIONS:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], linewidth=2)
    ax.scatter([pts[0, 0]], [pts[0, 1]], s=55)  # wrist
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.axis("off")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def draw_overlay(image_path: str, xy_norm: np.ndarray, out_path: str, flip_image: bool):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    if flip_image:
        img = cv2.flip(img, 1)  # horizontal flip

    h, w = img.shape[:2]

    # draw connections
    for a, b in HAND_CONNECTIONS:
        ax_, ay_ = xy_norm[a]
        bx_, by_ = xy_norm[b]
        p1 = (int(np.clip(ax_, 0.0, 1.0) * w), int(np.clip(ay_, 0.0, 1.0) * h))
        p2 = (int(np.clip(bx_, 0.0, 1.0) * w), int(np.clip(by_, 0.0, 1.0) * h))
        cv2.line(img, p1, p2, (0, 255, 0), 2)

    # draw points
    for (x, y) in xy_norm:
        px = int(np.clip(float(x), 0.0, 1.0) * w)
        py = int(np.clip(float(y), 0.0, 1.0) * h)
        cv2.circle(img, (px, py), 4, (0, 255, 0), -1)

    # wrist
    wx = int(np.clip(float(xy_norm[0, 0]), 0.0, 1.0) * w)
    wy = int(np.clip(float(xy_norm[0, 1]), 0.0, 1.0) * h)
    cv2.circle(img, (wx, wy), 7, (0, 0, 255), -1)

    cv2.imwrite(out_path, img)


def main():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Missing: {RAW_CSV}")
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(f"Missing: {CLEAN_CSV}")

    df_raw = pd.read_csv(RAW_CSV)
    df_clean = pd.read_csv(CLEAN_CSV)

    # choose sample from CLEAN (because it contains mirror_flipped)
    if FORCE_INSTANCE_ID is not None:
        chosen = df_clean[df_clean["instance_id"] == FORCE_INSTANCE_ID]
        if chosen.empty:
            raise ValueError(f"FORCE_INSTANCE_ID not found in clean CSV: {FORCE_INSTANCE_ID}")
        row_clean = chosen.iloc[0]
    else:
        row_clean = pick_best_row(df_clean, LABEL, REQUIRE_FLIPPED)

    instance_id = str(row_clean["instance_id"])
    flipped_flag = bool(row_clean["mirror_flipped"]) if "mirror_flipped" in row_clean.index else False

    print("Chosen instance_id:", instance_id)
    print("mirror_flipped:", flipped_flag)

    # find the matching RAW row (so overlay aligns with the original photo)
    raw_match = df_raw[df_raw["instance_id"] == instance_id]
    if raw_match.empty:
        raise ValueError("Could not find matching instance_id in landmarks_raw.csv")
    row_raw = raw_match.iloc[0]

    image_path = os.path.normpath(os.path.join(DATA_DIR, instance_id))
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

 
    # --- Build 4-panel from RAW so it matches the photo overlay ---
    raw_pts_for_panel = row_to_points_xyz(row_raw)          # raw landmarks from landmarks_raw.csv
    raw_xy, translated, scaled, _ = compute_steps(raw_pts_for_panel)

# Do NOT apply horizontal mirroring for visualisation
    mirrored = scaled.copy()


    # Symmetric limits so mirrored never clips
    pad = 0.15
    xmax = float(np.max(np.abs(translated[:, 0]))) + pad
    ymax = float(np.max(np.abs(translated[:, 1]))) + pad
    shared_xlim = (-xmax, xmax)
    shared_ylim = (-ymax, ymax)

    raw_xlim = (-0.05, 1.05)
    raw_ylim = (-0.05, 1.05)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    plot_points(axes[0], raw_xy, "Raw (from RAW CSV)", xlim=raw_xlim, ylim=raw_ylim)
    plot_points(axes[1], translated, "Translated (wrist at origin)", xlim=shared_xlim, ylim=shared_ylim)
    plot_points(axes[2], scaled, "Scaled (divide by max dist)", xlim=shared_xlim, ylim=shared_ylim)
    plot_points(axes[3], mirrored, "Mirrored (flip x in hand space)", xlim=shared_xlim, ylim=shared_ylim)

    out_panel = os.path.join(OUT_DIR, "preprocess_steps_4panel.png")
    plt.tight_layout()
    plt.savefig(out_panel, dpi=250, bbox_inches="tight")
    plt.close(fig)


    # Overlay uses RAW coordinates on the original image (this will match)
    raw_pts = row_to_points_xyz(row_raw)
    raw_xy_overlay = raw_pts[:, :2].astype(float)

    out_overlay_raw = os.path.join(OUT_DIR, "raw_landmarks_on_original_image.jpg")
    draw_overlay(image_path, raw_xy_overlay, out_overlay_raw, flip_image=False)

    # Optional: show CLEAN landmarks on a flipped image (only meaningful if mirror_flipped True)
    if SAVE_CLEAN_OVERLAY_ON_FLIPPED_IMAGE and flipped_flag:
        clean_xy_overlay = row_to_points_xyz(row_clean)[:, :2].astype(float)
        out_overlay_clean = os.path.join(OUT_DIR, "clean_landmarks_on_flipped_image.jpg")
        draw_overlay(image_path, clean_xy_overlay, out_overlay_clean, flip_image=True)
        print("Saved clean overlay on flipped image:", out_overlay_clean)

    print("\nSaved outputs:")
    print("4 panel figure:", out_panel)
    print("Raw overlay (correct alignment):", out_overlay_raw)


if __name__ == "__main__":
    main()
