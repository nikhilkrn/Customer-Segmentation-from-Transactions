
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def select_kmeans(X_train, k_range=range(2, 11), random_state=42):
    """
    Evaluate K-Means models using inertia and silhouette score.
    """

    results = []

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10
        )

        labels = model.fit_predict(X_train)

        results.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette Score": silhouette_score(X_train, labels)
        })

    return pd.DataFrame(results)


def train_kmeans(X_train, n_clusters=2, random_state=42):
    """
    Train the final K-Means model.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    model.fit(X_train)

    return model


def select_gmm(X_train,n_components=range(2, 11),covariance_types=("full", "tied", "diag", "spherical"),random_state=42):
    """
    Evaluate GMM configurations using BIC and AIC.
    """

    results = []

    for covariance in covariance_types:
        for k in n_components:

            model = GaussianMixture(
                n_components=k,
                covariance_type=covariance,
                random_state=random_state,
                n_init=3,
                reg_covar=1e-5
            )

            model.fit(X_train)

            results.append({
                "K": k,
                "Covariance": covariance,
                "BIC": model.bic(X_train),
                "AIC": model.aic(X_train)
            })

    return pd.DataFrame(results)


def train_gmm(X_train,n_components=10,covariance_type="full",random_state=42):
    """
    Train the final Gaussian Mixture Model.
    """

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=3,
        reg_covar=1e-5
    )

    model.fit(X_train)

    return model
