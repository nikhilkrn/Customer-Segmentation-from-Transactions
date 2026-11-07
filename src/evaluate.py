
import numpy as np
import pandas as pd

from itertools import combinations

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score
)

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def evaluate_clustering(X, labels):
    """Calculate clustering evaluation metrics."""

    return {
        "Silhouette Score": silhouette_score(X, labels),
        "Davies-Bouldin Index": davies_bouldin_score(X, labels)
    }


def compare_models(X_test, kmeans_labels, gmm_labels):
    """Compare K-Means and GMM on the test dataset."""

    kmeans_metrics = evaluate_clustering(X_test, kmeans_labels)
    gmm_metrics = evaluate_clustering(X_test, gmm_labels)

    comparison = pd.DataFrame([
        {"Model": "K-Means", **kmeans_metrics},
        {"Model": "GMM", **gmm_metrics}
    ])

    return comparison


def calculate_stability(X_train,model_type,seeds=range(10)):
    """Calculate pairwise ARI across different random seeds."""

    all_labels = []

    for seed in seeds:

        if model_type == "KMeans":

            model = KMeans(
                n_clusters=2,
                random_state=seed,
                n_init=10
            )

        elif model_type == "GMM":

            model = GaussianMixture(
                n_components=10,
                covariance_type="full",
                random_state=seed,
                n_init=3,
                reg_covar=1e-5
            )

        else:
            raise ValueError("Unsupported model type")

        labels = model.fit_predict(X_train)
        all_labels.append(labels)

    ari_scores = [
        adjusted_rand_score(all_labels[i], all_labels[j])
        for i, j in combinations(range(len(all_labels)), 2)
    ]

    return {
        "Model": model_type,
        "Mean ARI": np.mean(ari_scores),
        "ARI Std": np.std(ari_scores)
    }


def profile_clusters(rfm, labels):
    """Generate customer-level cluster profiles."""

    profile_data = rfm.copy()
    profile_data["Cluster"] = labels

    cluster_profile = profile_data.groupby("Cluster").agg({
        "CustomerID": "count",
        "Recency": ["mean", "median"],
        "Frequency": ["mean", "median"],
        "Monetary": ["mean", "median"]
    })

    return cluster_profile
