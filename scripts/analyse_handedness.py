# analyse_handedness_C_only.py
# Uses MediaPipe Hands to detect whether images for label C are left or right hand.
# Walks through ../data/C only, reads images, runs MediaPipe, and saves:
# 1) a full CSV of all C images + handedness
# 2) a CSV of only the images detected as RIGHT (so you can inspect them quickly)
# 3) a short text summary including the exact filenames detected as RIGHT

import os
import cv2
import pandas as pd
import mediapipe as mp

DATA_DIR = "../data"
LABEL = "C"
LABEL_DIR = os.path.join(DATA_DIR, LABEL)

OUT_CSV = "../outputs/handedness_analysis_C.csv"
RIGHT_ONLY_CSV = "../outputs/handedness_C_right_only.csv"
OUT_SUMMARY_TXT = "../outputs/handedness_summary_C.txt"

os.makedirs("../outputs", exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

# common image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

rows = []
failed = 0
total = 0

if not os.path.isdir(LABEL_DIR):
    raise FileNotFoundError(f"Could not find label folder: {LABEL_DIR}")

for fname in sorted(os.listdir(LABEL_DIR)):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in IMG_EXTS:
        continue

    total += 1
    path = os.path.join(LABEL_DIR, fname)

    bgr = cv2.imread(path)
    if bgr is None:
        failed += 1
        rows.append(
            {
                "label": LABEL,
                "file": fname,
                "path": path,
                "detected": False,
                "handedness": None,
                "score": None,
                "notes": "cv2_failed_to_read",
            }
        )
        continue

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks or not result.multi_handedness:
        failed += 1
        rows.append(
            {
                "label": LABEL,
                "file": fname,
                "path": path,
                "detected": False,
                "handedness": None,
                "score": None,
                "notes": "no_hand_detected",
            }
        )
        continue

    # for max_num_hands=1, take the first
    handed = result.multi_handedness[0].classification[0].label  # "Left" or "Right"
    score = float(result.multi_handedness[0].classification[0].score)

    rows.append(
        {
            "label": LABEL,
            "file": fname,
            "path": path,
            "detected": True,
            "handedness": handed,
            "score": score,
            "notes": "",
        }
    )

hands.close()

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

# right-hand only subset
df_right = df[(df["detected"] == True) & (df["handedness"] == "Right")].copy()
df_right.to_csv(RIGHT_ONLY_CSV, index=False)

# summary stats + exact filenames
summary_lines = []
summary_lines.append(f"Label scanned: {LABEL}")
summary_lines.append(f"Folder: {LABEL_DIR}")
summary_lines.append(f"Total images scanned: {total}")
summary_lines.append(f"Hands detected: {int(df['detected'].sum())}")
summary_lines.append(f"Detection failures: {failed}")

if df["detected"].any():
    overall = df[df["detected"]]["handedness"].value_counts()
    summary_lines.append("\nOverall handedness counts (C only):")
    summary_lines.append(overall.to_string())

summary_lines.append(f"\nRight-hand detected count (C only): {len(df_right)}")
summary_lines.append("Right-hand detected files (C only):")
if len(df_right) == 0:
    summary_lines.append("None")
else:
    for f in df_right["file"].tolist():
        summary_lines.append(f" - {f}")

with open(OUT_SUMMARY_TXT, "w") as f:
    f.write("\n".join(summary_lines) + "\n")

print("Saved C-only handedness CSV to:", OUT_CSV)
print("Saved C-only RIGHT detections CSV to:", RIGHT_ONLY_CSV)
print("Saved C-only summary to:", OUT_SUMMARY_TXT)
print("\n".join(summary_lines))
