# Customer Segmentation from Transactions

**Course:** EC-417 — Introduction to Machine Intelligence  
**Project:** Project 11 — Customer Segmentation from Transactions  
**Institution:** National Institute of Technology Andhra Pradesh  
**Team:** Nikhil Karn and Sumanth

## 1. Project overview

This project applies unsupervised machine learning to the UCI Online Retail dataset to identify groups of customers with similar purchasing behaviour.

Transaction-level records are aggregated into customer-level Recency, Frequency and Monetary (RFM) features. K-Means is used as the baseline clustering method and a Gaussian Mixture Model (GMM) is evaluated as an alternative. The models are compared using internal clustering metrics and stability across different random initialisations.

The objective is to investigate whether the resulting groups provide interpretable customer profiles that could support differentiated marketing strategies. The clusters are exploratory groupings, not known or verified customer categories.

## 2. Dataset

The project uses the **Online Retail** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail).

- **Dataset ID:** 352
- **Original format:** Excel (`.xlsx`)
- **Original records:** 541,909 transactions
- **Period:** 1 December 2010 to 9 December 2011
- **Fields:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`

The original dataset is not committed to this repository. Download it using:

```bash
python scripts/get_data.py
```

The script stores the Excel file at `data/raw/Online_Retail.xlsx`. See [`data/README.md`](data/README.md) for dataset details and handling notes.

## 3. Data preparation and RFM features

The transaction data was cleaned before customer aggregation. The implemented cleaning steps are:

1. Remove rows without a `CustomerID`.
2. Remove cancelled invoices whose invoice number starts with `C`.
3. Remove rows with non-positive `UnitPrice`.
4. Remove rows with non-positive `Quantity`.
5. Remove exact duplicate rows.

After these steps, the cleaned dataset contains **392,692 transactions and 4,338 unique customers**.

Missing product descriptions are not, by themselves, a reason to remove a transaction. Negative quantities and cancelled invoices are excluded, so the Monetary feature represents the value of retained positive-purchase transactions; it is not net spending after returns and refunds.

For each customer:

- **Recency:** Number of days between the customer's most recent retained purchase and the reference date (one day after the latest transaction in the dataset).
- **Frequency:** Number of distinct invoices associated with the customer.
- **Monetary:** Sum of `Quantity × UnitPrice` for the customer's retained transactions, in pounds sterling.

The RFM features are right-skewed. The modelling workflow applies `log1p` to the three features and standardises them. The customer records are split into training and test sets (80:20, random state 42). The scaler is fitted on the training set only and then applied to the test set.

## 4. Methods

### K-Means baseline

K-Means was evaluated for cluster counts from 2 to 10. Inertia (for the elbow method) and silhouette score were examined. The selected baseline uses **K=2**.

### Gaussian Mixture Model

GMM configurations with 2–10 components and `full`, `tied`, `diag`, and `spherical` covariance types were compared using Bayesian Information Criterion (BIC). The selected configuration was **10 components with full covariance**.

A lower BIC selects a configuration among the GMM candidates; it does not by itself show that GMM produces better customer segments than K-Means.

### Evaluation

The selected models were evaluated on the same held-out test customers using:

- **Silhouette score:** higher values generally indicate better separation and cohesion.
- **Davies–Bouldin index:** lower values generally indicate more compact and separated clusters.
- **Adjusted Rand Index (ARI):** used to compare cluster assignments across 10 random initialisations on the training data. The mean and standard deviation of pairwise ARI values are reported as a measure of run-to-run stability.

## 5. Results

The current recorded held-out test results are:

| Model | Configuration | Silhouette score | Davies–Bouldin index |
|---|---|---:|---:|
| K-Means | K=2 | 0.4331 | 0.8937 |
| GMM | 10 components, full covariance | 0.0842 | 2.3906 |

Stability across 10 random seeds:

| Model | Mean pairwise ARI | ARI standard deviation |
|---|---:|---:|
| K-Means | 1.0000 | 0.0000 |
| GMM | 0.8941 | 0.0571 |

On the held-out test set, K-Means has a higher silhouette score and a lower Davies–Bouldin index than GMM. K-Means was also more stable across the tested random initialisations. Therefore, K-Means is used for the customer segment profiles discussed in this project.

These results are internal clustering measures; they do not establish that the segments improve revenue or marketing outcomes. The ten-seed ARI analysis measures stability of assignments on the training data, not ten independent held-out test evaluations.

### Customer profiles

The K-Means customer profiles are:

| Segment | Customers | Mean recency (days) | Mean frequency (invoices) | Mean monetary (£) |
|---|---:|---:|---:|---:|
| Cluster 0 | 1,671 | 26.05 | 8.43 | 4,529.87 |
| Cluster 1 | 2,667 | 134.19 | 1.67 | 494.11 |

- **Cluster 0 — Recently active, higher-value customers:** these customers purchased more recently, placed more distinct invoices, and had higher retained purchase value on average.
- **Cluster 1 — Less recent, lower-frequency customers:** these customers had a longer time since their last purchase, fewer invoices, and lower retained purchase value on average.

### Potential marketing actions

The following are hypotheses suggested by the profiles, not experimentally validated outcomes:

- **Cluster 0:** consider loyalty rewards, early access, personalised recommendations, and retention-focused communication.
- **Cluster 1:** consider carefully targeted re-engagement campaigns, relevant product recommendations, and incentives for a repeat purchase.

Any campaign should be evaluated before claiming that it improves business performance.

## 6. Visualisation

The project uses Principal Component Analysis (PCA) to project the three standardised RFM features into two dimensions for visual inspection of the K-Means clusters. The first two components explain approximately **93.85%** of the variance in the transformed feature space (PC1: 75.11%; PC2: 18.74%).

The PCA scatter plot is a visual summary; it does not replace the quantitative evaluation metrics.

The generated figure is stored in `results/figures/kmeans_pca_clusters.png`.

## 7. Repository structure

```text
TermProject/
├── data/
│   ├── raw/                     # Downloaded source data (not committed)
│   ├── processed/               # Cleaned transaction data
│   └── README.md
├── notebooks/
│   ├── 01_dataExploration.ipynb
│   ├── 02_rfm_analysis.ipynb
│   └── 03_customer_segmentation.ipynb
├── results/
│   ├── figures/
│   ├── customer_rfm.csv
│   ├── customer_segments.csv
│   ├── cluster_profiles.csv
│   ├── kmeans_selection.csv
│   ├── gmm_selection.csv
│   ├── model_comparison.csv
│   ├── stability_results.csv
│   └── metrics.csv
├── scripts/
│   └── get_data.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── evaluate.py
│   └── train.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Some generated files may appear only after running the relevant notebook or pipeline. Do not commit the raw dataset, virtual environment, or serialized model checkpoints.

## 8. Installation and execution

### Requirements

- Python version compatible with the pinned packages in `requirements.txt`.
- Git.
- Internet access to download the UCI dataset.
- A terminal, and optionally VS Code/Jupyter to run the notebooks.

### Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows Git Bash:**

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Obtain the dataset

From the repository root:

```bash
python scripts/get_data.py
```

This downloads the original Excel file to `data/raw/Online_Retail.xlsx`.

### Run the notebooks

The notebooks document the exploratory and analytical workflow. Run them from top to bottom in this order:

1. `notebooks/01_dataExploration.ipynb`
2. `notebooks/02_rfm_analysis.ipynb`
3. `notebooks/03_customer_segmentation.ipynb`

The first notebook creates the cleaned transaction dataset, and the second creates the customer-level RFM dataset used by the segmentation analysis.

### Run the Python training pipeline

The current `src/train.py` expects the cleaned transaction CSV at:

```text
data/processed/online_retail_cleaned.csv
```

After generating that file through the data-cleaning notebook, run the pipeline from the repository root:

```bash
python -m src.train
```

The pipeline writes the model-selection tables, evaluation results, customer profiles, and related CSV outputs to `results/`.

**Reproducibility note:** the notebook-based cleaning step is currently required before running `python -m src.train`. A single command that downloads raw data, performs cleaning, trains the models, and regenerates every result has not yet been verified. Do not treat the workflow as a fully fresh-clone, one-command pipeline until that integration has been implemented and tested.



## 9. AI-use statement

AI assistance was used as a support tool during the project for clarifying machine-learning concepts, discussing implementation approaches, troubleshooting selected errors, and reviewing parts of the code and documentation. It also helped draft or refine some code sections and explanatory text.

---

**Project status:** Core RFM segmentation and model comparison have been implemented and executed. Repository integration, documentation, report, presentation, and final clean-clone verification should be completed before submission.
