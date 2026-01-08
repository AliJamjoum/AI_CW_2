# train_and_evaluate.py
# Part 2c main pipeline: load cleaned data, split, set up CV, optimise and evaluate supervised models.

import os  # standard library: used for creating folders and building file paths
import math  # standard library: used for sqrt and other maths in kNN
from collections import Counter  # standard library: majority vote counting

import pandas as pd  # allowed for data handling

from sklearn.model_selection import train_test_split, StratifiedKFold  # splitting + CV strategy
from sklearn.tree import DecisionTreeClassifier  # decision tree model
from sklearn.model_selection import cross_val_score  # runs CV scoring

# Evaluation metrics (sklearn)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.pipeline import Pipeline  # chaining preprocessing + model
from sklearn.preprocessing import StandardScaler  # scaling for Logistic Regression
from sklearn.linear_model import LogisticRegression  # third classifier


# -------------------------
# 1) Paths and reproducibility settings

SEED = 2025  # one place to control randomness for reproducibility

CLEAN_PATH = "../outputs/landmarks_clean.csv"  # where your cleaned features live
RESULTS_DIR = "../outputs/part2c_results"  # where results can be saved later (tables, plots, etc.)

os.makedirs(RESULTS_DIR, exist_ok=True)  # create results folder if it does not exist


# -------------------------
# 2) Load cleaned data

df = pd.read_csv(CLEAN_PATH)  # reads clean features into a dataframe

# 3) X is input features and y is labels
feature_cols = [c for c in df.columns if c.startswith("f")]  # selects feature columns f0..f62
X = df[feature_cols].values  # numpy array (allowed generally for the coursework pipeline)
y = df["label"].values  # array of labels (A..J)

# Derive labels dynamically to avoid hardcoding A..J
# This keeps confusion matrices correct even if a class is missing in a split.
labels = sorted(pd.Series(y).unique())


# -------------------------
# 4) Stratified train-test split
# test_size=0.2 means 20% test, 80% training
# stratify=y keeps the class balance similar in train and test
# random_state fixes randomness for reproducibility

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

# 5) Print split sizes
print("Total instances:", len(y))
print("Training instances:", len(y_train))
print("Test instances:", len(y_test))

# Print class distribution
train_counts = pd.Series(y_train).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

print("\nClass counts (train):")
print(train_counts.to_string())
print("\nClass counts (test):")
print(test_counts.to_string())


# -------------------------
# 5) CV setup
# 5-fold cross validation only applied to training set
# StratifiedKFold keeps class balance inside each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("\nCV setup: 5-fold StratifiedKFold with shuffle=True and random_state=", SEED)


# -------------------------
# Helper: Evaluate a fitted sklearn model cleanly (Decision Tree and Logistic Regression)
def report_fitted_model(name, model, X_train, y_train, X_test, y_test, labels):
    """
    Print train and test accuracy, balanced accuracy, macro metrics (precision/recall/F1),
    confusion matrix, and a per-class classification report.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Core metrics
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_bal_acc = balanced_accuracy_score(y_train, train_pred)
    test_bal_acc = balanced_accuracy_score(y_test, test_pred)

    # Macro averaged metrics (treat each class equally)
    test_macro_precision = precision_score(y_test, test_pred, average="macro", zero_division=0)
    test_macro_recall = recall_score(y_test, test_pred, average="macro", zero_division=0)  # macro sensitivity
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


# =========================================================
# DECISION TREE
# =========================================================

# Baseline Decision Tree with default settings
dt_default = DecisionTreeClassifier(random_state=SEED)

# CV baseline: print both accuracy and balanced accuracy
# n_jobs=-1 lets CV folds run in parallel (speeds up runtime)
dt_cv_bal = cross_val_score(dt_default, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
dt_cv_acc = cross_val_score(dt_default, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

print("\nDecision Tree (default) - 5-fold CV:")
print("Accuracy folds:", dt_cv_acc)
print("Accuracy mean:", dt_cv_acc.mean())
print("Accuracy std:", dt_cv_acc.std())
print("Balanced accuracy folds:", dt_cv_bal)
print("Balanced accuracy mean:", dt_cv_bal.mean())
print("Balanced accuracy std:", dt_cv_bal.std())

# Hyperparameters (kept compact and high value)
# NOTE: The full grid can be slow. This version is smaller but still meaningful.
max_depth_grid = [None, 10, 20]
min_samples_leaf_grid = [1, 5, 10]
min_samples_split_grid = [2, 10]
ccp_alpha_grid = [0.0, 1e-3, 1e-2]
max_features_grid = [None, "sqrt"]
criterion_grid = ["gini", "entropy"]
class_weight_grid = [None, "balanced"]

best_dt = None
best_params = None
best_cv_mean = -1.0  # track best mean CV balanced accuracy

combo_count = 0  # progress tracking

for depth in max_depth_grid:
    for leaf in min_samples_leaf_grid:
        for min_split in min_samples_split_grid:
            for alpha in ccp_alpha_grid:
                for max_feat in max_features_grid:
                    for criterion in criterion_grid:
                        for class_weight in class_weight_grid:

                            combo_count += 1
                            if combo_count % 50 == 0:
                                print("DT tuning progress - combinations tried:", combo_count)

                            dt = DecisionTreeClassifier(
                                random_state=SEED,
                                max_depth=depth,
                                min_samples_leaf=leaf,
                                min_samples_split=min_split,
                                ccp_alpha=alpha,
                                max_features=max_feat,
                                criterion=criterion,
                                class_weight=class_weight
                            )

                            # Use balanced accuracy for hyperparameter selection
                            scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
                            mean_score = scores.mean()

                            # Tie-break rule: if equal, prefer simpler tree
                            is_better = mean_score > best_cv_mean
                            if (not is_better) and (abs(mean_score - best_cv_mean) < 1e-12) and (best_params is not None):
                                current_depth = 10**9 if depth is None else depth
                                best_depth = 10**9 if best_params["max_depth"] is None else best_params["max_depth"]

                                current_simplicity = (current_depth, -leaf, -min_split, -alpha)
                                best_simplicity = (
                                    best_depth,
                                    -best_params["min_samples_leaf"],
                                    -best_params["min_samples_split"],
                                    -best_params["ccp_alpha"]
                                )
                                if current_simplicity < best_simplicity:
                                    is_better = True

                            if is_better:
                                best_cv_mean = mean_score
                                best_dt = dt
                                best_params = {
                                    "max_depth": depth,
                                    "min_samples_leaf": leaf,
                                    "min_samples_split": min_split,
                                    "ccp_alpha": alpha,
                                    "max_features": max_feat,
                                    "criterion": criterion,
                                    "class_weight": class_weight
                                }

print("\nDecision Tree (tuned) - best CV result (balanced accuracy):")
print("Best params:", best_params)
print("Best mean CV balanced accuracy:", best_cv_mean)

# Retrain best tree on full training set and evaluate
best_dt.fit(X_train, y_train)
report_fitted_model("Decision Tree", best_dt, X_train, y_train, X_test, y_test, labels)


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

# Baseline Logistic Regression pipeline
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

# Hyperparameters for Logistic Regression
# Hyperparameters for Logistic Regression
# C controls regularisation strength (small C means stronger regularisation)
# class_weight can help if some classes are rarer
# NOTE: In sklearn >= 1.8, 'penalty' is deprecated. We rely on default L2 penalty.
C_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
class_weight_grid = [None, "balanced"]

best_lr = None
best_lr_params = None
best_lr_cv_mean = -1.0  # best mean CV balanced accuracy

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
report_fitted_model("Logistic Regression", best_lr, X_train, y_train, X_test, y_test, labels)

# =========================================================
# kNN (manual implementation, standard library only)
# =========================================================
# Coursework requirement: kNN must be implemented from scratch using only Python standard built-in libraries.
# To keep this very clear, the kNN section below avoids sklearn and numpy operations inside the kNN logic.
#
# Important: kNN is distance-based, so scaling matters.
# We implement standardisation manually using training-set mean and std computed with standard libraries.

def compute_standardisation_params(X_rows):
    """
    Compute per-feature mean and std from X_rows (list of lists).
    Standard library only.

    Returns:
    means: list[float]
    stds: list[float] with small epsilon protection for zero-variance features
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
        if stds[j] == 0.0:
            stds[j] = 1e-12  # protect against divide-by-zero

    return means, stds


def standardise_dataset(X_rows, means, stds):
    """Apply standardisation: (x - mean) / std, standard library only."""
    scaled = []
    for row in X_rows:
        new_row = []
        for j in range(len(row)):
            new_row.append((float(row[j]) - means[j]) / stds[j])
        scaled.append(new_row)
    return scaled


# Convert training and test data into plain Python lists for strict standard library kNN pipeline
X_train_list = X_train.tolist()
X_test_list = X_test.tolist()
y_train_list = list(y_train)
y_test_list = list(y_test)

# Fit scaler on training set only, then apply to both train and test
knn_means, knn_stds = compute_standardisation_params(X_train_list)
X_train_scaled = standardise_dataset(X_train_list, knn_means, knn_stds)
X_test_scaled = standardise_dataset(X_test_list, knn_means, knn_stds)


def euclidean_distance(a, b):
    """Standard Euclidean distance, standard library only."""
    total = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        total += diff * diff
    return math.sqrt(total)


def manhattan_distance(a, b):
    """Manhattan distance, sometimes useful as an alternative metric."""
    total = 0.0
    for i in range(len(a)):
        total += abs(a[i] - b[i])
    return total


def knn_predict_single(x_test, X_train_rows, y_train_rows, k, weighting="uniform", metric="euclidean"):
    """
    Predict one label for a single test point using kNN.
    weighting: "uniform" or "distance"
    metric: "euclidean" or "manhattan"
    """
    distances = []

    # choose distance function
    if metric == "euclidean":
        dist_fn = euclidean_distance
    else:
        dist_fn = manhattan_distance

    # compute distance to every training point
    for i in range(len(X_train_rows)):
        d = dist_fn(x_test, X_train_rows[i])
        distances.append((d, y_train_rows[i]))

    # sort by distance
    distances.sort(key=lambda t: t[0])
    neighbours = distances[:k]

    if weighting == "uniform":
        votes = [label for _, label in neighbours]
        return Counter(votes).most_common(1)[0][0]

    # distance-weighted vote: closer neighbours have more influence
    weights = {}
    for d, label in neighbours:
        w = 1.0 / (d + 1e-8)  # avoid division by zero
        weights[label] = weights.get(label, 0.0) + w
    return max(weights.items(), key=lambda t: t[1])[0]


def knn_predict_dataset(X_test_rows, X_train_rows, y_train_rows, k, weighting, metric):
    """Predict labels for a dataset using the manual kNN."""
    predictions = []
    for x in X_test_rows:
        predictions.append(knn_predict_single(x, X_train_rows, y_train_rows, k, weighting, metric))
    return predictions


def simple_accuracy(y_true, y_pred):
    """Manual accuracy: correct / total, standard library only."""
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


def balanced_accuracy_manual(y_true, y_pred, all_labels):
    """
    Manual balanced accuracy:
    compute recall per class and average them.
    Standard library only.
    """
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


def per_class_counts(y_true, y_pred, all_labels):
    """
    Compute TP, FP, FN per class using standard library only.
    Returns dict: label -> (tp, fp, fn)
    """
    counts = {}
    for lab in all_labels:
        counts[lab] = [0, 0, 0]  # tp, fp, fn

    for i in range(len(y_true)):
        true_lab = y_true[i]
        pred_lab = y_pred[i]

        if pred_lab == true_lab:
            counts[true_lab][0] += 1  # tp
        else:
            counts[pred_lab][1] += 1  # fp
            counts[true_lab][2] += 1  # fn

    return {lab: (counts[lab][0], counts[lab][1], counts[lab][2]) for lab in all_labels}


def macro_precision_recall_f1_manual(y_true, y_pred, all_labels):
    """
    Compute macro precision, macro recall, macro F1 using standard library only.
    """
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


# Hyperparameters for manual kNN
k_grid = [1, 3, 5, 7, 9, 11]
weighting_grid = ["uniform", "distance"]
metric_grid = ["euclidean", "manhattan"]

best_knn_params = None
best_knn_cv_mean = -1.0  # best mean CV balanced accuracy

for k in k_grid:
    for weighting in weighting_grid:
        for metric in metric_grid:
            fold_scores = []

            for train_idx, val_idx in cv.split(X_train_scaled, y_train_list):
                X_tr = [X_train_scaled[i] for i in train_idx]
                y_tr = [y_train_list[i] for i in train_idx]
                X_val = [X_train_scaled[i] for i in val_idx]
                y_val = [y_train_list[i] for i in val_idx]

                preds = knn_predict_dataset(X_val, X_tr, y_tr, k, weighting, metric)

                bal = balanced_accuracy_manual(y_val, preds, labels)
                fold_scores.append(bal)

            mean_score = sum(fold_scores) / len(fold_scores)

            is_better = mean_score > best_knn_cv_mean
            if (not is_better) and (abs(mean_score - best_knn_cv_mean) < 1e-12) and (best_knn_params is not None):
                if k < best_knn_params["k"]:
                    is_better = True

            if is_better:
                best_knn_cv_mean = mean_score
                best_knn_params = {"k": k, "weighting": weighting, "metric": metric}

print("\nkNN (tuned) - best CV result (balanced accuracy):")
print("Best params:", best_knn_params)
print("Best mean CV balanced accuracy:", best_knn_cv_mean)

best_k = best_knn_params["k"]
best_weighting = best_knn_params["weighting"]
best_metric = best_knn_params["metric"]

# Train and test predictions for kNN
knn_train_pred = knn_predict_dataset(X_train_scaled, X_train_scaled, y_train_list, best_k, best_weighting, best_metric)
knn_test_pred = knn_predict_dataset(X_test_scaled, X_train_scaled, y_train_list, best_k, best_weighting, best_metric)

# Core metrics
knn_train_acc = simple_accuracy(y_train_list, knn_train_pred)
knn_test_acc = simple_accuracy(y_test_list, knn_test_pred)

knn_train_bal = balanced_accuracy_manual(y_train_list, knn_train_pred, labels)
knn_test_bal = balanced_accuracy_manual(y_test_list, knn_test_pred, labels)

# Additional macro metrics (manual)
knn_macro_p, knn_macro_r, knn_macro_f1 = macro_precision_recall_f1_manual(y_test_list, knn_test_pred, labels)

print("\nkNN (best) performance:")
print("Train accuracy:", knn_train_acc)
print("Test accuracy:", knn_test_acc)
print("Train balanced accuracy:", knn_train_bal)
print("Test balanced accuracy:", knn_test_bal)

print("\nAdditional test metrics (macro averaged):")
print("Macro precision:", knn_macro_p)
print("Macro recall (macro sensitivity):", knn_macro_r)
print("Macro F1:", knn_macro_f1)

# Confusion matrix for kNN
knn_cm = confusion_matrix(y_test_list, knn_test_pred, labels=labels)
print("\nkNN confusion matrix (test):")
print(knn_cm)
