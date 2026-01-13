import matplotlib.pyplot as plt

# Configuration
n_folds = 5
train_color = "#1f4fd8"   # dark blue
val_color = "#9ec9ff"     # light blue

fig, ax = plt.subplots(figsize=(8, 4))

# Draw folds
for fold in range(n_folds):
    for block in range(n_folds):
        if block == fold:
            color = val_color
            label = "Validation" if fold == 0 else ""
        else:
            color = train_color
            label = "Training" if fold == 0 else ""

        ax.add_patch(
            plt.Rectangle(
                (block, n_folds - fold - 1),
                1,
                0.8,
                facecolor=color,
                edgecolor="white",
                label=label
            )
        )

# Formatting
ax.set_xlim(0, n_folds)
ax.set_ylim(0, n_folds)
ax.set_xticks(range(n_folds))
ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
ax.set_yticks(range(n_folds))
ax.set_yticklabels([f"Run {i+1}" for i in range(n_folds)][::-1])

ax.set_title("Stratified 5-Fold Cross-Validation", fontsize=14)
ax.set_xlabel("Data folds")
ax.set_ylabel("Cross-validation runs")

# Legend (avoid duplicates)
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), loc="lower right")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("stratified_5_fold_cv.png", dpi=200)
plt.show()
