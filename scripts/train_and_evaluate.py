

import os  
import math  
from collections import Counter  

import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.model_selection import cross_val_score  
from sklearn.tree import DecisionTreeClassifier  


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression 




SEED = 2025  

CLEAN_PATH = "../outputs/landmarks_clean.csv"
RESULTS_DIR = "../outputs/part2c_results"  

os.makedirs(RESULTS_DIR, exist_ok=True)  




def save_confusion_matrix_png(cm, labels, title, out_path, normalise=False):
    """
    Save a confusion matrix as a PNG figure.
    normalise=False shows raw counts.
    normalise=True shows row-normalised proportions (recall-style).
    """
    plt.figure(figsize=(7, 6))

    cm_to_plot = cm.astype(float)
    if normalise:
        row_sums = cm_to_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_to_plot = cm_to_plot / row_sums

    plt.imshow(cm_to_plot)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if normalise:
                txt = f"{cm_to_plot[i, j]:.2f}"
            else:
                txt = str(int(cm[i, j]))
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_model_comparison_bar(values, names, ylabel, title, out_path, ylim=(0, 1)):
    """Save a simple bar chart for quick model comparison."""
    plt.figure(figsize=(7, 4))
    plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_results_summary_csv(rows, out_path):
    """Save a CSV summary of model metrics."""
    pd.DataFrame(rows).to_csv(out_path, index=False)


def model_to_filename_prefix(name: str) -> str:
    """
    Turn a model name into a simple filename prefix.
    No extra helper needed elsewhere; keep it local and obvious.
    """
    return name.lower().replace(" ", "_")


def save_false_negative_analysis(cm, labels, model_name, results_dir, top_k=3):
    """
    False negatives for class i = all true i that were predicted as NOT i.
    From confusion matrix:
      FN_i = row_sum_i - TP_i
      Recall_i = TP_i / (TP_i + FN_i) = TP_i / row_sum_i  (if row_sum_i > 0)

    Saves a CSV for your report/poster and prints worst classes.
    """
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1)
    tp = np.diag(cm)
    fn = row_sums - tp

    recall = np.zeros(len(labels), dtype=float)
    for i in range(len(labels)):
        if row_sums[i] > 0:
            recall[i] = tp[i] / row_sums[i]
        else:
            recall[i] = 0.0

    
    df_fn = pd.DataFrame({
        "label": labels,
        "true_count": row_sums.astype(int),
        "true_positives": tp.astype(int),
        "false_negatives": fn.astype(int),
        "recall": recall
    })

    
    prefix = model_to_filename_prefix(model_name)
    out_csv = os.path.join(results_dir, f"{prefix}_false_negatives_per_class.csv")
    df_fn.to_csv(out_csv, index=False)

 
    worst_by_fn = df_fn.sort_values(["false_negatives", "true_count"], ascending=[False, False]).head(top_k)
    worst_by_recall = df_fn.sort_values(["recall", "true_count"], ascending=[True, False]).head(top_k)

    print(f"\n{model_name} false negative analysis (per class):")
    print(f"Saved per-class FN/recall table to: {out_csv}")

    print(f"\nWorst classes by false negatives (top {top_k}):")
    print(worst_by_fn.to_string(index=False))

    print(f"\nWorst classes by recall (top {top_k}):")
    print(worst_by_recall.to_string(index=False))

    return df_fn




df = pd.read_csv(CLEAN_PATH)  


feature_cols = [c for c in df.columns if c.startswith("f")] 
X = df[feature_cols].values  
y = df["label"].values  

labels = sorted(pd.Series(y).unique())




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

print("Total instances:", len(y))
print("Training instances:", len(y_train))
print("Test instances:", len(y_test))


train_counts = pd.Series(y_train).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

print("\nClass counts (train):")
print(train_counts.to_string())
print("\nClass counts (test):")
print(test_counts.to_string())




cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("\nCV setup: 5-fold StratifiedKFold with shuffle=True and random_state=", SEED)


def report_fitted_model(name, model, X_train, y_train, X_test, y_test, labels, results_dir):
    """
    Print train and test accuracy, balanced accuracy, macro metrics (precision/recall/F1),
    confusion matrix, and a per-class classification report.
    Also saves confusion matrix figures and returns a metrics dict for summary tables.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_bal_acc = balanced_accuracy_score(y_train, train_pred)
    test_bal_acc = balanced_accuracy_score(y_test, test_pred)

    test_macro_precision = precision_score(y_test, test_pred, average="macro", zero_division=0)
    test_macro_recall = recall_score(y_test, test_pred, average="macro", zero_division=0)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)

    print(f"\n{name} (best) performance:")
    print("Train accuracy:", train_acc)
    print("Test accuracy:", test_acc)
    print("Train balanced accuracy:", train_bal_acc)
    print("Test balanced accuracy:", test_bal_acc)

    print("\nAdditional test metrics (macro averaged):")
    print("Macro precision:", test_macro_precision)
    print("Macro recall (macro sensitivity):", test_macro_recall)
    print("Macro F1:", test_macro_f1)

    cm = confusion_matrix(y_test, test_pred, labels=labels)
    print(f"\n{name} confusion matrix (test):")
    print(cm)

    print("\nPer class report (test):")
    print(classification_report(y_test, test_pred, labels=labels, zero_division=0))


    prefix = model_to_filename_prefix(name)
    save_confusion_matrix_png(
        cm, labels,
        title=f"{name} Confusion Matrix (Test, counts)",
        out_path=os.path.join(results_dir, f"{prefix}_cm_test_counts.png"),
        normalise=False
    )
    save_confusion_matrix_png(
        cm, labels,
        title=f"{name} Confusion Matrix (Test, normalised)",
        out_path=os.path.join(results_dir, f"{prefix}_cm_test_normalised.png"),
        normalise=True
    )

 
    save_false_negative_analysis(cm, labels, name, results_dir, top_k=3)

    return {
        "model": name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_balanced_accuracy": train_bal_acc,
        "test_balanced_accuracy": test_bal_acc,
        "test_macro_precision": test_macro_precision,
        "test_macro_recall": test_macro_recall,
        "test_macro_f1": test_macro_f1,
    }


# ========================================================
# DECISION TREE
# ========================================================

dt_default = DecisionTreeClassifier(random_state=SEED)

dt_cv_bal = cross_val_score(dt_default, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
dt_cv_acc = cross_val_score(dt_default, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

print("\nDecision Tree (default) - 5-fold CV:")
print("Accuracy folds:", dt_cv_acc)
print("Accuracy mean:", dt_cv_acc.mean())
print("Accuracy std:", dt_cv_acc.std())
print("Balanced accuracy folds:", dt_cv_bal)
print("Balanced accuracy mean:", dt_cv_bal.mean())
print("Balanced accuracy std:", dt_cv_bal.std())


max_depth_grid = [None, 10, 20]
min_samples_leaf_grid = [1, 5, 10]
class_weight_grid = [None, "balanced"]
criterion_fixed = "gini"

best_dt = None
best_dt_params = None
best_dt_cv_mean = -1.0

combo_count = 0

for depth in max_depth_grid:
    for leaf in min_samples_leaf_grid:
        for class_weight in class_weight_grid:

            combo_count += 1
            if combo_count % 10 == 0:
                print("DT tuning progress - combinations tried:", combo_count)

            dt = DecisionTreeClassifier(
                random_state=SEED,
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight=class_weight,
                criterion=criterion_fixed
            )

            scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
            mean_score = scores.mean()

         
            is_better = mean_score > best_dt_cv_mean
            if (not is_better) and (abs(mean_score - best_dt_cv_mean) < 1e-12) and (best_dt_params is not None):
                current_depth = 10**9 if depth is None else depth
                best_depth = 10**9 if best_dt_params["max_depth"] is None else best_dt_params["max_depth"]

                current_simplicity = (current_depth, -leaf)
                best_simplicity = (best_depth, -best_dt_params["min_samples_leaf"])

                if current_simplicity < best_simplicity:
                    is_better = True

            if is_better:
                best_dt_cv_mean = mean_score
                best_dt = dt
                best_dt_params = {
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "class_weight": class_weight,
                    "criterion": criterion_fixed
                }

print("\nDecision Tree (tuned) - best CV result (balanced accuracy):")
print("Best params:", best_dt_params)
print("Best mean CV balanced accuracy:", best_dt_cv_mean)

best_dt.fit(X_train, y_train)
dt_metrics = report_fitted_model("Decision Tree", best_dt, X_train, y_train, X_test, y_test, labels, RESULTS_DIR)


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

lr_default = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=SEED,
        solver="lbfgs"
    ))
])

lr_cv_bal = cross_val_score(lr_default, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
lr_cv_acc = cross_val_score(lr_default, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

print("\nLogistic Regression (default) - 5-fold CV:")
print("Accuracy folds:", lr_cv_acc)
print("Accuracy mean:", lr_cv_acc.mean())
print("Accuracy std:", lr_cv_acc.std())
print("Balanced accuracy folds:", lr_cv_bal)
print("Balanced accuracy mean:", lr_cv_bal.mean())
print("Balanced accuracy std:", lr_cv_bal.std())

C_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
class_weight_grid = [None, "balanced"]

best_lr = None
best_lr_params = None
best_lr_cv_mean = -1.0

for C in C_grid:
    for class_weight in class_weight_grid:
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=C,
                class_weight=class_weight,
                solver="lbfgs",
                max_iter=2000,
                random_state=SEED
            ))
        ])

        scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
        mean_score = scores.mean()

        if mean_score > best_lr_cv_mean:
            best_lr_cv_mean = mean_score
            best_lr = lr
            best_lr_params = {"C": C, "class_weight": class_weight}

print("\nLogistic Regression (tuned) - best CV result (balanced accuracy):")
print("Best params:", best_lr_params)
print("Best mean CV balanced accuracy:", best_lr_cv_mean)

best_lr.fit(X_train, y_train)
lr_metrics = report_fitted_model("Logistic Regression", best_lr, X_train, y_train, X_test, y_test, labels, RESULTS_DIR)


# ====================================================
# kNN
# ====================================================

# this function computes the mean and std for each feature
# it only uses the training data to avoid data leakage
def compute_standardisation_params(X_rows):
    """
    compute per-feature mean and std from training rows
    this is needed because knn is distance based
    """
    n = len(X_rows)
    d = len(X_rows[0])

    means = [0.0] * d
    for row in X_rows:
        for j in range(d):
            means[j] += float(row[j])
    for j in range(d):
        means[j] /= n

    stds = [0.0] * d
    for row in X_rows:
        for j in range(d):
            diff = float(row[j]) - means[j]
            stds[j] += diff * diff
    for j in range(d):
        stds[j] = math.sqrt(stds[j] / n)
        # prevent divide by zero
        if stds[j] == 0.0:
            stds[j] = 1e-12

    return means, stds


# apply (x - mean) / std to each feature
# this puts all features on the same scale
def standardise_dataset(X_rows, means, stds):
    """standardise dataset using training means and stds"""
    scaled = []
    for row in X_rows:
        new_row = []
        for j in range(len(row)):
            new_row.append((float(row[j]) - means[j]) / stds[j])
        scaled.append(new_row)
    return scaled


# convert numpy arrays to python lists for manual looping
X_train_list = X_train.tolist()
X_test_list = X_test.tolist()
y_train_list = list(y_train)
y_test_list = list(y_test)

# compute scaling parameters from training data only
knn_means, knn_stds = compute_standardisation_params(X_train_list)

# scale training and test sets using training statistics
X_train_scaled = standardise_dataset(X_train_list, knn_means, knn_stds)
X_test_scaled = standardise_dataset(X_test_list, knn_means, knn_stds)


# euclidean distance (straight line distance)
def euclidean_distance(a, b):
    total = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        total += diff * diff
    return math.sqrt(total)


# manhattan distance (sum of absolute differences)
def manhattan_distance(a, b):
    total = 0.0
    for i in range(len(a)):
        total += abs(a[i] - b[i])
    return total


# predict a single test sample using knn
# supports uniform voting or distance weighted voting
def knn_predict_single(x_test, X_train_rows, y_train_rows,
                       k, weighting="uniform", metric="euclidean"):

    distances = []

    # choose distance function
    if metric == "euclidean":
        dist_fn = euclidean_distance
    else:
        dist_fn = manhattan_distance

    # compute distance to every training sample
    for i in range(len(X_train_rows)):
        d = dist_fn(x_test, X_train_rows[i])
        distances.append((d, y_train_rows[i]))

    # sort neighbours by distance and keep k closest
    distances.sort(key=lambda t: t[0])
    neighbours = distances[:k]

    # uniform voting just counts labels
    if weighting == "uniform":
        votes = [label for _, label in neighbours]
        return Counter(votes).most_common(1)[0][0]

    # distance weighting gives closer neighbours more influence
    weights = {}
    for d, label in neighbours:
        w = 1.0 / (d + 1e-8)
        weights[label] = weights.get(label, 0.0) + w

    return max(weights.items(), key=lambda t: t[1])[0]


# run knn prediction for a full dataset
def knn_predict_dataset(X_test_rows, X_train_rows, y_train_rows,
                        k, weighting, metric):
    predictions = []
    for x in X_test_rows:
        predictions.append(
            knn_predict_single(x, X_train_rows, y_train_rows, k, weighting, metric)
        )
    return predictions


# simple accuracy = correct predictions / total
def simple_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


# balanced accuracy = average recall across all classes
def balanced_accuracy_manual(y_true, y_pred, all_labels):
    recalls = []

    for lab in all_labels:
        tp = 0
        fn = 0
        for i in range(len(y_true)):
            if y_true[i] == lab:
                if y_pred[i] == lab:
                    tp += 1
                else:
                    fn += 1
        denom = tp + fn
        if denom == 0:
            continue
        recalls.append(tp / denom)

    if len(recalls) == 0:
        return 0.0
    return sum(recalls) / len(recalls)


# count tp, fp and fn for each class
def per_class_counts(y_true, y_pred, all_labels):
    counts = {}
    for lab in all_labels:
        counts[lab] = [0, 0, 0]  # tp, fp, fn

    for i in range(len(y_true)):
        true_lab = y_true[i]
        pred_lab = y_pred[i]

        if pred_lab == true_lab:
            counts[true_lab][0] += 1
        else:
            counts[pred_lab][1] += 1
            counts[true_lab][2] += 1

    return {lab: (counts[lab][0], counts[lab][1], counts[lab][2]) for lab in all_labels}


# compute macro precision, recall and f1
def macro_precision_recall_f1_manual(y_true, y_pred, all_labels):
    counts = per_class_counts(y_true, y_pred, all_labels)

    precisions = []
    recalls = []
    f1s = []

    for lab in all_labels:
        tp, fp, fn = counts[lab]

        denom_p = tp + fp
        precision = (tp / denom_p) if denom_p != 0 else 0.0

        denom_r = tp + fn
        if denom_r == 0:
            continue
        recall = tp / denom_r

        denom_f1 = precision + recall
        f1 = (2 * precision * recall / denom_f1) if denom_f1 != 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    if len(recalls) == 0:
        return 0.0, 0.0, 0.0

    return (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f1s) / len(f1s)
    )


# baseline knn configuration before tuning
BASELINE_K = 5
BASELINE_WEIGHTING = "uniform"
BASELINE_METRIC = "euclidean"

# run 5-fold cv on training set only for baseline knn
baseline_fold_scores = []

for train_idx, val_idx in cv.split(X_train_scaled, y_train_list):
    X_tr = [X_train_scaled[i] for i in train_idx]
    y_tr = [y_train_list[i] for i in train_idx]
    X_val = [X_train_scaled[i] for i in val_idx]
    y_val = [y_train_list[i] for i in val_idx]

    preds = knn_predict_dataset(
        X_val, X_tr, y_tr,
        BASELINE_K, BASELINE_WEIGHTING, BASELINE_METRIC
    )

    bal = balanced_accuracy_manual(y_val, preds, labels)
    baseline_fold_scores.append(bal)

baseline_knn_cv_mean = sum(baseline_fold_scores) / len(baseline_fold_scores)
baseline_knn_cv_std = (
    sum((s - baseline_knn_cv_mean) ** 2 for s in baseline_fold_scores)
    / len(baseline_fold_scores)
) ** 0.5


# grids for tuning knn hyperparameters
k_grid = [1, 3, 5, 7, 9, 11]
weighting_grid = ["uniform", "distance"]
metric_grid = ["euclidean", "manhattan"]

best_knn_params = None
best_knn_cv_mean = -1.0

# grid search using balanced accuracy
for k in k_grid:
    for weighting in weighting_grid:
        for metric in metric_grid:

            fold_scores = []

            for train_idx, val_idx in cv.split(X_train_scaled, y_train_list):
                X_tr = [X_train_scaled[i] for i in train_idx]
                y_tr = [y_train_list[i] for i in train_idx]
                X_val = [X_train_scaled[i] for i in val_idx]
                y_val = [y_train_list[i] for i in val_idx]

                preds = knn_predict_dataset(
                    X_val, X_tr, y_tr, k, weighting, metric
                )
                bal = balanced_accuracy_manual(y_val, preds, labels)
                fold_scores.append(bal)

            mean_score = sum(fold_scores) / len(fold_scores)

            # tie break: if scores are equal, prefer smaller k
            is_better = mean_score > best_knn_cv_mean
            if (not is_better) and abs(mean_score - best_knn_cv_mean) < 1e-12 and best_knn_params is not None:
                if k < best_knn_params["k"]:
                    is_better = True

            if is_better:
                best_knn_cv_mean = mean_score
                best_knn_params = {
                    "k": k,
                    "weighting": weighting,
                    "metric": metric
                }


# evaluate best knn on train and test sets
best_k = best_knn_params["k"]
best_weighting = best_knn_params["weighting"]
best_metric = best_knn_params["metric"]

# training accuracy
knn_train_pred = knn_predict_dataset(
    X_train_scaled, X_train_scaled, y_train_list,
    best_k, best_weighting, best_metric
)

# test accuracy using training set as reference
knn_test_pred = knn_predict_dataset(
    X_test_scaled, X_train_scaled, y_train_list,
    best_k, best_weighting, best_metric
)

knn_train_acc = simple_accuracy(y_train_list, knn_train_pred)
knn_test_acc = simple_accuracy(y_test_list, knn_test_pred)

knn_train_bal = balanced_accuracy_manual(y_train_list, knn_train_pred, labels)
knn_test_bal = balanced_accuracy_manual(y_test_list, knn_test_pred, labels)

knn_macro_p, knn_macro_r, knn_macro_f1 = macro_precision_recall_f1_manual(
    y_test_list, knn_test_pred, labels
)

print("\nkNN (best) performance:")
print("Train accuracy:", knn_train_acc)
print("Test accuracy:", knn_test_acc)
print("Train balanced accuracy:", knn_train_bal)
print("Test balanced accuracy:", knn_test_bal)

print("\nAdditional test metrics (macro averaged):")
print("Macro precision:", knn_macro_p)
print("Macro recall (macro sensitivity):", knn_macro_r)
print("Macro F1:", knn_macro_f1)

knn_cm = confusion_matrix(y_test_list, knn_test_pred, labels=labels)
print("\nkNN confusion matrix (test):")
print(knn_cm)

save_confusion_matrix_png(
    knn_cm, labels,
    title="kNN Confusion Matrix (Test, counts)",
    out_path=os.path.join(RESULTS_DIR, "knn_cm_test_counts.png"),
    normalise=False
)
save_confusion_matrix_png(
    knn_cm, labels,
    title="kNN Confusion Matrix (Test, normalised)",
    out_path=os.path.join(RESULTS_DIR, "knn_cm_test_normalised.png"),
    normalise=True
)


save_false_negative_analysis(knn_cm, labels, "kNN", RESULTS_DIR, top_k=3)

knn_metrics = {
    "model": "kNN",
    "train_accuracy": knn_train_acc,
    "test_accuracy": knn_test_acc,
    "train_balanced_accuracy": knn_train_bal,
    "test_balanced_accuracy": knn_test_bal,
    "test_macro_precision": knn_macro_p,
    "test_macro_recall": knn_macro_r,
    "test_macro_f1": knn_macro_f1,
}


summary_rows = [dt_metrics, lr_metrics, knn_metrics]
save_results_summary_csv(summary_rows, os.path.join(RESULTS_DIR, "supervised_results_summary.csv"))

names = [r["model"] for r in summary_rows]
test_accs = [r["test_accuracy"] for r in summary_rows]
test_bal_accs = [r["test_balanced_accuracy"] for r in summary_rows]
macro_f1s = [r["test_macro_f1"] for r in summary_rows]

save_model_comparison_bar(
    test_accs, names,
    ylabel="Test accuracy",
    title="Supervised model comparison: Test accuracy",
    out_path=os.path.join(RESULTS_DIR, "compare_test_accuracy.png"),
    ylim=(0, 1)
)

save_model_comparison_bar(
    test_bal_accs, names,
    ylabel="Test balanced accuracy",
    title="Supervised model comparison: Test balanced accuracy",
    out_path=os.path.join(RESULTS_DIR, "compare_test_balanced_accuracy.png"),
    ylim=(0, 1)
)

save_model_comparison_bar(
    macro_f1s, names,
    ylabel="Test macro F1",
    title="Supervised model comparison: Test macro F1",
    out_path=os.path.join(RESULTS_DIR, "compare_test_macro_f1.png"),
    ylim=(0, 1)
)

print("\nSaved figures and summary CSV to:", RESULTS_DIR)
