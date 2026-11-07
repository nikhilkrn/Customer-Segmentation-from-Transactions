

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load the cleaned transaction dataset."""
    data = pd.read_csv(file_path)
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    return data


def create_rfm(data):
    """Generate customer-level Recency, Frequency and Monetary features."""

    data = data.copy()

    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    reference_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)

    recency = (data.groupby("CustomerID")["InvoiceDate"].max().reset_index())

    recency["Recency"] = (reference_date - recency["InvoiceDate"]).dt.days

    recency = recency[["CustomerID", "Recency"]]

    frequency = (data.groupby("CustomerID")["InvoiceNo"].nunique().reset_index().rename(columns={"InvoiceNo": "Frequency"})
    )

    monetary = (data.groupby("CustomerID")["TotalPrice"].sum().reset_index().rename(columns={"TotalPrice": "Monetary"}))

    rfm = recency.merge(frequency, on="CustomerID")
    rfm = rfm.merge(monetary, on="CustomerID")

    rfm["CustomerID"] = rfm["CustomerID"].astype("int64")

    return rfm



def preprocess_rfm(rfm, random_state=42, test_size=0.2):
    """Transform, split and standardize RFM features."""

    features = rfm[["Recency", "Frequency", "Monetary"]].copy()

    # Log transformation
    log_features = np.log1p(features)

    # Split dataset
    X_train, X_test = train_test_split(log_features,test_size=test_size,random_state=random_state)

    # Standardization
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "train_indices": X_train.index,
        "test_indices": X_test.index,
        "scaler": scaler,
        "log_features": log_features
    }
