# cluster_and_analyse.py
# Part 2d: Unsupervised clustering on MediaPipe landmark features.
# Follows lab guidance: z-score normalisation + sklearn KMeans + cluster plots. :contentReference[oaicite:4]{index=4}
# Also runs hierarchical clustering and evaluates alignment with true labels (analysis only).

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
#uses z score scaling, kmeans and ward linkage use distances
from sklearn.cluster import KMeans, AgglomerativeClustering
#agglomerative is ward linkage (heirarchal clustering)
from sklearn.decomposition import PCA
#PCA reduces dimensionality for 2d plots
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    # ^ compare cluster assignments to true lables
    silhouette_score,
    #measures cluster separation using only X and prediced clusters
)

SEED = 2025
#makes kmeans initialisation deterministic

CLEAN_PATH = "../outputs/landmarks_clean.csv"
RESULTS_DIR = "../outputs/part2d_results"
SUPERVISED_SUMMARY_PATH = "../outputs/part2c_results/supervised_results_summary.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)


#given a 2D embedding (PCA) and produces scatter plot
def save_scatter_2d(X_2d, colors, title, out_path):
    plt.figure(figsize=(7, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=12, alpha=0.85, c=colors)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

#plots contigency table as image. counts is matrix where rows are clusters and columns r true lables
def save_heatmap(counts, xlabels, ylabels, title, out_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(counts, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    plt.yticks(range(len(ylabels)), ylabels)
    plt.xlabel("True label")
    plt.ylabel("Cluster")
    plt.title(title)

    # diagnostic - ideally each cluster row has one dominant column
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            plt.text(j, i, str(int(counts[i, j])), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


#purity: for each cluster, take size of majority true class inside that cluster, sum the majority counts and divide by N
def cluster_purity_and_table(y_true, y_cluster):
    """
    Purity = sum over clusters of max class count in that cluster / N
    Also returns contingency table (clusters x labels) for heatmap plotting.
    """
    labels = sorted(pd.Series(y_true).unique())
    clusters = sorted(pd.Series(y_cluster).unique())

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}

    cont = np.zeros((len(clusters), len(labels)), dtype=int)
    for yt, yc in zip(y_true, y_cluster):
        cont[cluster_to_idx[yc], label_to_idx[yt]] += 1

    purity = cont.max(axis=1).sum() / len(y_true) if len(y_true) else 0.0
    return purity, cont, clusters, labels


def read_best_supervised_if_exists():
    if not os.path.exists(SUPERVISED_SUMMARY_PATH):
        return None
    try:
        df = pd.read_csv(SUPERVISED_SUMMARY_PATH)
        if "test_balanced_accuracy" not in df.columns:
            return None
        best = df.sort_values("test_balanced_accuracy", ascending=False).iloc[0].to_dict()
        return best
    except Exception:
        return None


# -------------------------
# 1) Load cleaned data (remove label for clustering)
df = pd.read_csv(CLEAN_PATH)

feature_cols = [c for c in df.columns if c.startswith("f")]
X = df[feature_cols].values
y = df["label"].values

labels_sorted = sorted(pd.Series(y).unique())
k = len(labels_sorted)

print("Loaded:", CLEAN_PATH)
print("Instances:", len(y))
print("Features:", len(feature_cols))
print("Classes:", labels_sorted)
print("Using k =", k)

# -------------------------
# 2) Z-score normalisation (lab guideline) :contentReference[oaicite:5]{index=5}
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 3) PCA for visualisation (2D)
pca = PCA(n_components=2, random_state=SEED)
X_2d = pca.fit_transform(X_scaled)

label_to_id = {lab: i for i, lab in enumerate(labels_sorted)}
y_ids = np.array([label_to_id[lab] for lab in y])

save_scatter_2d(
    X_2d,
    y_ids,
    "PCA (2D) of landmark features coloured by true label",
    os.path.join(RESULTS_DIR, "pca_true_labels.png"),
)

# -------------------------
# 4) K-means clustering (sklearn) :contentReference[oaicite:6]{index=6}
# Lab example uses n_init=20; we follow that for consistency. :contentReference[oaicite:7]{index=7}
kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=20)
km_clusters = kmeans.fit_predict(X_scaled)

km_purity, km_cont, km_cluster_ids, _ = cluster_purity_and_table(y, km_clusters)

km_ari = adjusted_rand_score(y, km_clusters)
km_nmi = normalized_mutual_info_score(y, km_clusters)
km_h, km_c, km_v = homogeneity_completeness_v_measure(y, km_clusters)
km_sil = silhouette_score(X_scaled, km_clusters)

print("\nK-means (k=10) effectiveness:")
print("Purity:", round(km_purity, 4))
print("ARI:", round(km_ari, 4))
print("NMI:", round(km_nmi, 4))
print("Homogeneity / Completeness / V-measure:", round(km_h, 4), round(km_c, 4), round(km_v, 4))
print("Silhouette:", round(km_sil, 4))

save_scatter_2d(
    X_2d,
    km_clusters,
    "K-means clusters on PCA projection",
    os.path.join(RESULTS_DIR, "pca_kmeans_clusters.png"),
)

save_heatmap(
    km_cont,
    labels_sorted,
    km_cluster_ids,
    "K-means: cluster vs true label counts",
    os.path.join(RESULTS_DIR, "kmeans_cluster_label_heatmap.png"),
)

# -------------------------
# 5) Hierarchical clustering (Agglomerative)
agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
agg_clusters = agg.fit_predict(X_scaled)

agg_purity, agg_cont, agg_cluster_ids, _ = cluster_purity_and_table(y, agg_clusters)

agg_ari = adjusted_rand_score(y, agg_clusters)
agg_nmi = normalized_mutual_info_score(y, agg_clusters)
agg_h, agg_c, agg_v = homogeneity_completeness_v_measure(y, agg_clusters)
agg_sil = silhouette_score(X_scaled, agg_clusters)

print("\nHierarchical (Agglomerative, ward) effectiveness:")
print("Purity:", round(agg_purity, 4))
print("ARI:", round(agg_ari, 4))
print("NMI:", round(agg_nmi, 4))
print("Homogeneity / Completeness / V-measure:", round(agg_h, 4), round(agg_c, 4), round(agg_v, 4))
print("Silhouette:", round(agg_sil, 4))

save_scatter_2d(
    X_2d,
    agg_clusters,
    "Hierarchical clusters on PCA projection",
    os.path.join(RESULTS_DIR, "pca_hierarchical_clusters.png"),
)

save_heatmap(
    agg_cont,
    labels_sorted,
    agg_cluster_ids,
    "Hierarchical: cluster vs true label counts",
    os.path.join(RESULTS_DIR, "hierarchical_cluster_label_heatmap.png"),
)

# -------------------------
# 6) Save results summary CSV
summary = pd.DataFrame(
    [
        {
            "method": f"KMeans(k={k})",
            "purity": km_purity,
            "ARI": km_ari,
            "NMI": km_nmi,
            "homogeneity": km_h,
            "completeness": km_c,
            "v_measure": km_v,
            "silhouette": km_sil,
        },
        {
            "method": f"Agglomerative(ward, k={k})",
            "purity": agg_purity,
            "ARI": agg_ari,
            "NMI": agg_nmi,
            "homogeneity": agg_h,
            "completeness": agg_c,
            "v_measure": agg_v,
            "silhouette": agg_sil,
        },
    ]
)

out_csv = os.path.join(RESULTS_DIR, "unsupervised_results_summary.csv")
summary.to_csv(out_csv, index=False)

print("\nSaved unsupervised figures + summary CSV to:", RESULTS_DIR)
print("Summary CSV:", out_csv)

# -------------------------
# 7) Compare to best supervised model (from Part 2c CSV if present)
best_sup = read_best_supervised_if_exists()
if best_sup is None:
    print("\nSupervised summary not found (optional):", SUPERVISED_SUMMARY_PATH)
else:
    print("\nBest supervised model (Part 2c):")
    print("Model:", best_sup.get("model"))
    print("Test balanced accuracy:", round(float(best_sup.get("test_balanced_accuracy")), 4))
