# train_and_evaluate.py
# Part 2c main pipeline: load cleaned data, split, set up CV, and prepare evaluation.

import os  # standard library: used for creating folders and building file paths
import pandas as pd  # allowed for data handling
from sklearn.model_selection import train_test_split, StratifiedKFold  # splitting + CV strategy

from sklearn.tree import DecisionTreeClassifier  # the decision tree model
from sklearn.model_selection import cross_val_score  # runs CV scoring easily
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix  # evaluation metrics

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# 1) Paths
CLEAN_PATH = "../outputs/landmarks_clean.csv"  # where your cleaned features live
RESULTS_DIR = "../outputs/part2c_results"  # where we will save results (tables, plots later)

os.makedirs(RESULTS_DIR, exist_ok=True)  # create results folder if it does not exist

# 2) Load cleaned data
df = pd.read_csv(CLEAN_PATH)  # reads the clean features into dataframe

# 3) x is input features and y is labels
feature_cols = [c for c in df.columns if c.startswith("f")]  # selects all feature values
X = df[feature_cols].values  # turns data frame into numpy array 
y = df["label"].values  # selects all labels and turns it into an array (a,b,c...)

# 4)stratified test train split
# test_size=0.2 means 20% test, 80% training
# stratify=y keeps the class balance similar in train and test
# random_state fixes randomness for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, #feature matrix (samples x 63)
    y, # label vector (number of samples)
    test_size=0.2,
    stratify=y, #maintains class distribution 
    random_state=2025 #fixes randomness of the split 
)

# 5)prints split sizes 
print("total instances:", len(y)) 
print("training instances:", len(y_train))
print("test instances:", len(y_test))

#converts label into a pandas series and counts how many times each label appears + ordered alphabetically
train_counts = pd.Series(y_train).value_counts().sort_index()  
test_counts = pd.Series(y_test).value_counts().sort_index()  # counts each label in test

#prints class distribution
print("\nClass counts (train):")
print(train_counts.to_string())
print("\nClass counts (test):")
print(test_counts.to_string())

# 5 fold cross validation - ONLY APPLIED TO TRAINING
# StratifiedKFold - splits training data into folds, preserves class balance 
# n_splits = 5, 5 total, each fold used once as validation
# shuffle = True, randomises order before splitting 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

print("\nCV setup: 5-fold StratifiedKFold with shuffle=True and random_state=2025")

#DECISION TREE

#creates decision tree classifier with default settings, random state so its reproducible
dt_default = DecisionTreeClassifier(random_state = 2025)

#runs cross validation to estimate performance - only on training data
#trains on diff subsets, evaluates robustness + better estimate of real performance
dt_cv_acc = cross_val_score(dt_default, X_train, y_train, cv=cv, scoring="accuracy")

#performance accuracies
print("\nDecision Tree (default) - 5-fold CV accuracy:")
print("Fold accuracies:", dt_cv_acc)
print("Mean CV accuracy:", dt_cv_acc.mean())
print("Std CV accuracy:", dt_cv_acc.std())
# mean shows average performance, std shows stability across folds.


# These control model complexity:
# max_depth limits how deep the tree can grow.
# min_samples_leaf forces each leaf to contain at least this many samples.
max_depth_grid = [None, 5, 10, 15, 20]
min_samples_leaf_grid = [1, 2, 5, 10]

#placeholder for best dt model 
best_dt = None
#placeholder for best params
best_params = None
#tracker for best cv score seen, -1 guaranteed to be worse than any valid accuracy
best_cv_mean = -1.0

#iterates over every combination of max depth and min samples leaf
for depth in max_depth_grid:
    for leaf in min_samples_leaf_grid:
        #creates decision tree using current depth and leaf size
        dt = DecisionTreeClassifier(
            random_state=2025,
            max_depth=depth,
            min_samples_leaf=leaf
        )
        #trains and evaluates specific tree configuration using cv, run 5 times with 4 training fold and 1 test fold
        scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring="accuracy")
        mean_score = scores.mean()
        #compresses fold level results into one number

        #saves stuff when better model is found
        if mean_score > best_cv_mean:
            best_cv_mean = mean_score
            best_dt = dt
            best_params = {"max_depth": depth, "min_samples_leaf": leaf}

#reports outcome of grid search
print("\nDecision Tree (tuned) - best CV result:")
print("Best params:", best_params)
print("Best mean CV accuracy:", best_cv_mean)

# Retrain the best tree on the full training set
best_dt.fit(X_train, y_train)
# fit trains the tree using all training data, using the chosen best hyperparameters.

# Evaluate on train and test
train_pred = best_dt.predict(X_train)
test_pred = best_dt.predict(X_test)
# predict produces class predictions for each instance.

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
# accuracy_score = correct predictions / total predictions

train_bal_acc = balanced_accuracy_score(y_train, train_pred)
test_bal_acc = balanced_accuracy_score(y_test, test_pred)
# balanced accuracy averages recall across classes, useful even if classes become imbalanced.

print("\nDecision Tree (best) performance:")
print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)
print("Train balanced accuracy:", train_bal_acc)
print("Test balanced accuracy:", test_bal_acc)

cm = confusion_matrix(y_test, test_pred, labels=list("ABCDEFGHIJ"))
# confusion_matrix shows which classes are being confused with others.
# labels fixes consistent ordering A..J.

print("\nDecision Tree confusion matrix (test):")
print(cm)


# LOGISTIC REGRESSION 

#creates base model that standardises features and trains logisitic regression
#pipeline makes sklearn object, pass a list of steps (each one is a tuple)
#scaler is the standardscaler object, model is label and logisticregression is classifier
lr_default = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=2025))
])
# StandardScaler is needed because Logistic Regression is sensitive to feature scale.
# max_iter is increased to avoid convergence warnings.

#computes 5 fold cv accuracy for base model, outputs a numpy array of 5 numbers (one per fold)
lr_cv_acc = cross_val_score(lr_default, X_train, y_train, cv=cv, scoring="accuracy")

#prints results
print("\nLogistic Regression (default) - 5-fold CV accuracy:")
print("Fold accuracies:", lr_cv_acc)
print("Mean CV accuracy:", lr_cv_acc.mean())
print("Std CV accuracy:", lr_cv_acc.std())

#hyperparameters list: regularisation strength and penalty type
#inverse of regularisation strength, small means strong and large is weak. (small penalises large weights - stops overfitting)
C_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
#how to penalise large weights (l2 is sum of squared weights)
penalty_grid = ["l2"]

#should rare classes matter more?
class_weight_grid = [None, "balanced"]


#best pipelines found so far
best_lr = None
best_lr_params = None
best_lr_cv_mean = -1.0

#loops over every combination               
for C in C_grid:
    for penalty in penalty_grid:
        for class_weight in class_weight_grid:
        #builds a pipeline 
            lr = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    C=C,
                    penalty=penalty,
                    class_weight=class_weight,
                    solver="lbfgs", #chooses optimisation algorithm
                    max_iter=2000,
                    random_state=2025
                ))
            ])

            #computes mean cv accuracy to summarise its performance
            scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring="accuracy")
            mean_score = scores.mean()

            #finds best performer
            if mean_score > best_lr_cv_mean:
                best_lr_cv_mean = mean_score
                best_lr = lr
                #dictionary storing chosen hyperparameters
                best_lr_params = {"C": C, "penalty": penalty, "class_weight": class_weight}


#prints best result
print("\nLogistic Regression (tuned) - best CV result:")
print("Best params:", best_lr_params)
print("Best mean CV accuracy:", best_lr_cv_mean)

#retrain best Logistic Regression on full training set
best_lr.fit(X_train, y_train)

#evaluate on the train and test data
lr_train_pred = best_lr.predict(X_train)
lr_test_pred = best_lr.predict(X_test)

lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)

lr_train_bal_acc = balanced_accuracy_score(y_train, lr_train_pred)
lr_test_bal_acc = balanced_accuracy_score(y_test, lr_test_pred)

print("\nLogistic Regression (best) performance:")
print("Train accuracy:", lr_train_acc)
print("Test accuracy:", lr_test_acc)
print("Train balanced accuracy:", lr_train_bal_acc)
print("Test balanced accuracy:", lr_test_bal_acc)

lr_cm = confusion_matrix(y_test, lr_test_pred, labels=list("ABCDEFGHIJ"))
print("\nLogistic Regression confusion matrix (test):")
print(lr_cm)

# -------------------------
# kNN (manual implementation)

# for a test point, compute distance to every training point, sort distances. 
# take the k nearest neighbours - predict by majority vote or distance weighted vote


import math
from collections import Counter

def euclidean_distance(a, b):
    #calculates euclidean distance for all feature vectors, close in space means similar class
    total = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        total += diff * diff
    return math.sqrt(total)

#x train and y train are reference sest, k is number of neighbours, weighting selects voting method
def knn_predict_single(x_test, X_train, y_train, k, weighting="uniform"):
    distances = []

    #computes distance to every training point 
    for i in range(len(X_train)):
        d = euclidean_distance(x_test, X_train[i])
        distances.append((d, y_train[i]))

    #sort by distance (sorts by first element in each tuple)
    distances.sort(key=lambda t: t[0])
    #takes the first k entries (closest training points)
    neighbours = distances[:k]

    #extracts only labels from k neighbours
    if weighting == "uniform":
        #list comprehension, _ ignores the distance and label keeps the class
        votes = [label for _, label in neighbours]
        #uses counter to count how many times label appears, most_common gives top 1 class as a list of tuples, so u get the letter 'A'
        return Counter(votes).most_common(1)[0][0]

    #closer neighbours more influence
    #computes weight w = 1/(distance+small no) add up the weights for each class
    elif weighting == "distance":
        weights = {}
        for d, label in neighbours:
            w = 1.0 / (d + 1e-8)
            #returns existing weight sum for that label or 0
            weights[label] = weights.get(label, 0.0) + w
        #weights.items() gives pairs like ('A', 2.7)
        #chooses pair with largest weight value and returns the letter
        return max(weights.items(), key=lambda t: t[1])[0]

#runs single point predictor for every test point, returns a list of predicted labels
def knn_predict_dataset(X_test, X_train, y_train, k, weighting):
    """Predict labels for a dataset."""
    predictions = []
    for x in X_test:
        pred = knn_predict_single(x, X_train, y_train, k, weighting)
        predictions.append(pred)
    return predictions

#odd k reduced tie likelihood
k_grid = [1, 3, 5, 7, 9, 11]
weighting_grid = ["uniform", "distance"]

#stores best settings
best_knn = None
best_knn_params = None
best_knn_cv_mean = -1.0

#outer loops over hyperparameters, will collect one accuracy per fold
for k in k_grid:
    for weighting in weighting_grid:
        #stores 5 accuracy values 
        fold_scores = []

        #indices for stratified folds
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr = [X_train[i] for i in train_idx]
            y_tr = [y_train[i] for i in train_idx]
            X_val = [X_train[i] for i in val_idx]
            y_val = [y_train[i] for i in val_idx]

            preds = knn_predict_dataset(X_val, X_tr, y_tr, k, weighting)
            acc = accuracy_score(y_val, preds)
            fold_scores.append(acc)

        mean_score = sum(fold_scores) / len(fold_scores)

        if mean_score > best_knn_cv_mean:
            best_knn_cv_mean = mean_score
            best_knn = (k, weighting)
            best_knn_params = {"k": k, "weighting": weighting}

print("\nkNN (tuned) - best CV result:")
print("Best params:", best_knn_params)
print("Best mean CV accuracy:", best_knn_cv_mean)

best_k = best_knn_params["k"]
best_weighting = best_knn_params["weighting"]

knn_train_pred = knn_predict_dataset(X_train, X_train, y_train, best_k, best_weighting)
knn_test_pred = knn_predict_dataset(X_test, X_train, y_train, best_k, best_weighting)

knn_train_acc = accuracy_score(y_train, knn_train_pred)
knn_test_acc = accuracy_score(y_test, knn_test_pred)

knn_train_bal_acc = balanced_accuracy_score(y_train, knn_train_pred)
knn_test_bal_acc = balanced_accuracy_score(y_test, knn_test_pred)

print("\nkNN (best) performance:")
print("Train accuracy:", knn_train_acc)
print("Test accuracy:", knn_test_acc)
print("Train balanced accuracy:", knn_train_bal_acc)
print("Test balanced accuracy:", knn_test_bal_acc)

knn_cm = confusion_matrix(y_test, knn_test_pred, labels=list("ABCDEFGHIJ"))
print("\nkNN confusion matrix (test):")
print(knn_cm)
