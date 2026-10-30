# Dataset: Online Retail

## Overview

The **Online Retail** dataset is obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail).

It contains transactional data from a UK-based online retailer covering the period from **1 December 2010 to 9 December 2011**. The dataset consists of 541,909 transaction records and includes information about invoices, products, quantities, prices, customers, and countries.

This dataset is used in our project to perform **customer segmentation using RFM analysis and clustering algorithms**.

## Dataset Source

- **Name:** Online Retail
- **Source:** UCI Machine Learning Repository
- **Dataset ID:** 352
- **Original link:** https://archive.ics.uci.edu/dataset/352/online+retail
- **File format:** Excel (`.xlsx`)
- **File used:** `Online_Retail.xlsx`

## Dataset Attributes

| Attribute | Description |
|---|---|
| InvoiceNo | Invoice number identifying a transaction |
| StockCode | Product code |
| Description | Product description |
| Quantity | Number of units purchased |
| InvoiceDate | Date and time of the transaction |
| UnitPrice | Price per unit in sterling (£) |
| CustomerID | Unique customer identifier |
| Country | Country where the customer resides |

## Data Handling

The dataset is stored locally in the `data/raw/` directory.

- The original dataset is kept unchanged.
- Records without a `CustomerID` will be excluded from customer segmentation.
- Cancelled invoices will be identified and handled during preprocessing.
- Missing values, returns, and unusual transaction values will be examined before final cleaning.

## Usage

The dataset is used to calculate the following customer-level RFM features:

- **Recency:** Time since the customer's most recent purchase.
- **Frequency:** Number of purchases made by the customer.
- **Monetary:** Total amount spent by the customer.

These features will subsequently be used for customer segmentation using K-Means and Gaussian Mixture Models (GMM).

## Data Availability

The dataset is not included in the GitHub repository. Users should download it directly from the original UCI source and place the Excel file in:

`data/raw/Online_Retail.xlsx`

