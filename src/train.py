
from pathlib import Path
import pandas as pd
import joblib

from .data import load_data, create_rfm, preprocess_rfm

from .models import (select_kmeans,train_kmeans,select_gmm,train_gmm)

from .evaluate import (compare_models,calculate_stability,profile_clusters)


RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Processed" / "online_retail_cleaned.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("\nStarting Customer Segmentation Pipeline")

    print("\n1. Loading cleaned transaction data...")

    data = load_data(DATA_PATH)

    print("Transaction dataset shape:", data.shape)

    print("\n2. Generating RFM features...")

    rfm = create_rfm(data)

    rfm.to_csv(RESULTS_DIR / "customer_rfm.csv",index=False)

    print("RFM dataset shape:", rfm.shape)

    print("\n3. Preprocessing RFM features...")

    preprocessing = preprocess_rfm(rfm,random_state=RANDOM_STATE,test_size=0.2)

    X_train = preprocessing["X_train_scaled"]
    X_test = preprocessing["X_test_scaled"]

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)


    print("\n4. Running K-Means model selection...")

    kmeans_results = select_kmeans(X_train,random_state=RANDOM_STATE)

    kmeans_results.to_csv(RESULTS_DIR / "kmeans_selection.csv",index=False)

    print(kmeans_results)



    print("\n5. Running GMM model selection...")

    gmm_results = select_gmm(X_train,random_state=RANDOM_STATE)

    gmm_results.to_csv(RESULTS_DIR / "gmm_selection.csv",index=False)

    print(gmm_results.sort_values("BIC").head(10))



    print("\n6. Training selected models...")

    kmeans = train_kmeans(X_train,n_clusters=2,random_state=RANDOM_STATE)

    gmm = train_gmm(X_train,n_components=10,covariance_type="full",random_state=RANDOM_STATE)

    print("K-Means training completed.")
    print("GMM training completed.")
    print("GMM converged:", gmm.converged_)

    joblib.dump(kmeans, MODEL_DIR / "kmeans_model.joblib")
    joblib.dump(gmm, MODEL_DIR / "gmm_model.joblib")

    joblib.dump(preprocessing["scaler"],MODEL_DIR / "scaler.joblib")

    print("\nTrained models and scaler saved successfully.")

    print("\n7. Predicting customer clusters...")

    kmeans_test_labels = kmeans.predict(X_test)
    gmm_test_labels = gmm.predict(X_test)

    kmeans_train_labels = kmeans.predict(X_train)
    gmm_train_labels = gmm.predict(X_train)


    print("\n8. Evaluating models...")

    comparison = compare_models(X_test,kmeans_test_labels,gmm_test_labels)

    comparison.to_csv(RESULTS_DIR / "model_comparison.csv",index=False)

    print(comparison)



    print("\n9. Calculating clustering stability...")

    stability_results = pd.DataFrame([calculate_stability(X_train, "KMeans"),calculate_stability(X_train, "GMM")])

    stability_results.to_csv(RESULTS_DIR / "stability_results.csv",index=False)

    print(stability_results)


    metrics_records = []


    for _, row in comparison.iterrows():

        metrics_records.append({"Model": row["Model"],"Seed": RANDOM_STATE,"Metric": "Silhouette Score","Value": row["Silhouette Score"]})

        metrics_records.append({"Model": row["Model"],"Seed": RANDOM_STATE,"Metric": "Davies-Bouldin Index","Value": row["Davies-Bouldin Index"]})


    # Stability metrics across 10 random seeds
    for _, row in stability_results.iterrows():

        metrics_records.append({"Model": row["Model"],"Seed": "0-9","Metric": "Mean ARI","Value": row["Mean ARI"]})

        metrics_records.append({"Model": row["Model"],"Seed": "0-9","Metric": "ARI Standard Deviation","Value": row["ARI Std"]})


    # Create and save consolidated metrics
    metrics_df = pd.DataFrame(metrics_records)

    metrics_df.to_csv(RESULTS_DIR / "metrics.csv",index=False)

    print("\nConsolidated metrics saved:")
    print(metrics_df)


    print("\n10. Generating customer profiles...")

    rfm_profile = rfm.copy()

    rfm_profile["Cluster"] = pd.NA

    rfm_profile.loc[preprocessing["train_indices"], "Cluster"] = kmeans_train_labels

    rfm_profile.loc[preprocessing["test_indices"], "Cluster"] = kmeans_test_labels

    cluster_profile = profile_clusters(rfm_profile,rfm_profile["Cluster"])

    rfm_profile.to_csv(RESULTS_DIR / "customer_segments.csv",index=False)

    cluster_profile.to_csv(RESULTS_DIR / "cluster_profiles.csv")

    print(cluster_profile)

    print("\nCustomer Segmentation Pipeline Completed!")


if __name__ == "__main__":
    main()
