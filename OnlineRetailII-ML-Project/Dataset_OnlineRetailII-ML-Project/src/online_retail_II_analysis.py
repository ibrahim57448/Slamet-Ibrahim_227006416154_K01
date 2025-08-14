#!/usr/bin/env python3
"""
Online Retail II - End-to-end analysis script

Features:
1) Data loading & cleaning from Excel (.xlsx)
2) Descriptive EDA with saved charts (PNG)
3) RFM-based customer segmentation using K-Means
4) Invoice-level revenue prediction with RandomForestRegressor
5) All artifacts saved under outputs/

Usage:
python src/online_retail_II_analysis.py --data_path "/path/to/online_retail_II.xlsx" --out_dir "outputs"

Notes:
- Uses only matplotlib for plots (no seaborn)
- Charts are saved as colored defaults by matplotlib
"""

import argparse
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

def load_data(data_path: str) -> pd.DataFrame:
    ext = str(data_path).lower()
    if ext.endswith(".xlsx") or ext.endswith(".xls"):
        # engine openpyxl is needed for .xlsx
        df = pd.read_excel(data_path, engine="openpyxl")
    elif ext.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    # Normalize column names (strip spaces, title-case like UCI version)
    df.columns = [c.strip() for c in df.columns]
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Expected core columns in Online Retail II
    # Some versions: 'Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'
    # Ensure columns exist (best-effort)
    colmap = {c.lower(): c for c in df.columns}
    def find(name):
        # return matching column (case-insensitive)
        for c in df.columns:
            if c.lower() == name.lower():
                return c
        # try partials
        for c in df.columns:
            if name.lower() in c.lower():
                return c
        return None

    invoice_col = find("Invoice") or find("InvoiceNo") or find("InvoiceNo.")
    stock_col = find("StockCode")
    desc_col = find("Description")
    qty_col = find("Quantity")
    date_col = find("InvoiceDate")
    price_col = find("UnitPrice")
    cust_col = find("CustomerID") or find("Customer ID")
    country_col = find("Country")

    required = [invoice_col, stock_col, qty_col, date_col, price_col]
    if any(col is None for col in required):
        raise ValueError("Dataset does not have expected columns. Required at least: Invoice, StockCode, Quantity, InvoiceDate, UnitPrice.")

    df = df.copy()
    # Parse date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    # Numerical sanity
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[qty_col, price_col])
    # Filter cancellations and non-positive values
    if invoice_col is not None:
        df = df[~df[invoice_col].astype(str).str.startswith("C")]
    df = df[(df[qty_col] > 0) & (df[price_col] > 0)]
    # Total price
    df["TotalPrice"] = df[qty_col] * df[price_col]
    # Standardize column names into a consistent set for later use
    df = df.rename(columns={
        invoice_col: "Invoice",
        stock_col: "StockCode",
        desc_col if desc_col else "Description": "Description",
        qty_col: "Quantity",
        date_col: "InvoiceDate",
        price_col: "UnitPrice",
    })
    if cust_col:
        df = df.rename(columns={cust_col: "CustomerID"})
    if country_col:
        df = df.rename(columns={country_col: "Country"})
    # Ensure types
    if "CustomerID" in df.columns:
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    if "Country" in df.columns:
        df["Country"] = df["Country"].astype(str)
    return df

def ensure_out(out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out

def plot_save(fig, out_path: Path):
    # Save figure and close
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)

def basic_eda(df: pd.DataFrame, out_dir: Path):
    # Monthly revenue
    s = df.set_index("InvoiceDate").resample("MS")["TotalPrice"].sum().dropna()
    fig = plt.figure(figsize=(9, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(s.index, s.values)
    ax.set_title("Monthly Revenue")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.grid(True, linestyle="--", alpha=0.4)
    plot_save(fig, out_dir / "monthly_revenue.png")

    # Top 10 products by revenue
    if "Description" in df.columns:
        top_products = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10)
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        ax.barh(top_products.index[::-1], top_products.values[::-1])
        ax.set_title("Top 10 Products by Revenue")
        ax.set_xlabel("Revenue")
        ax.set_ylabel("Product")
        plot_save(fig, out_dir / "top10_products.png")

    # Revenue by country (top 10)
    if "Country" in df.columns:
        top_countries = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        ax.bar(top_countries.index, top_countries.values)
        ax.set_title("Top 10 Countries by Revenue")
        ax.set_xlabel("Country")
        ax.set_ylabel("Revenue")
        ax.tick_params(axis='x', rotation=45, ha='right')
        plot_save(fig, out_dir / "top10_countries.png")

def rfm_kmeans(df: pd.DataFrame, out_dir: Path):
    # Require CustomerID for RFM
    if "CustomerID" not in df.columns:
        return None

    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (ref_date - x.max()).days,
        "Invoice": "nunique",
        "TotalPrice": "sum"
    }).rename(columns={"InvoiceDate": "Recency", "Invoice": "Frequency", "TotalPrice": "Monetary"})
    rfm = rfm[(rfm["Recency"] >= 0) & (rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)]

    # Log transform to reduce skew
    X = np.log1p(rfm[["Recency", "Frequency", "Monetary"]].values)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Choose k by best silhouette (k=2..6)
    best_k, best_score = None, -1
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        if score > best_score:
            best_k, best_score, best_model, best_labels = k, score, km, labels

    rfm["Cluster"] = best_labels
    rfm_path = out_dir / "rfm_segments.csv"
    rfm.to_csv(rfm_path, index=True)

    # Scatter plot (Recency vs Monetary) colored by cluster
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    for c in sorted(rfm["Cluster"].unique()):
        sub = rfm[rfm["Cluster"] == c]
        ax.scatter(sub["Recency"], sub["Monetary"], label=f"Cluster {c}", s=18)
    ax.set_title(f"RFM Segmentation (k={best_k})")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary (sum)")
    ax.legend()
    plot_save(fig, out_dir / "rfm_scatter.png")

    # Save meta
    meta = {"best_k": int(best_k), "silhouette": float(best_score)}
    with open(out_dir / "rfm_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"best_k": best_k, "silhouette": best_score, "path": str(rfm_path)}

def invoice_value_regression(df: pd.DataFrame, out_dir: Path):
    # Build invoice-level dataset
    gcols = ["Invoice"]
    agg = df.groupby(gcols).agg({
        "InvoiceDate": "min",
        "Quantity": "sum",
        "UnitPrice": ["mean", "max", "min"],
        "StockCode": "nunique",
        "TotalPrice": "sum"
    })
    agg.columns = ["InvoiceDate", "ItemCount", "AvgUnitPrice", "MaxUnitPrice", "MinUnitPrice", "InvoiceValue"]
    X = agg[["ItemCount", "AvgUnitPrice", "MaxUnitPrice", "MinUnitPrice", "StockCode_nunique" if "StockCode_nunique" in agg.columns else ""]].copy()
    # Clean accidental empty-string column name if created
    if "" in X.columns:
        X = X.drop(columns=[""])
    # Add unique products
    if "StockCode_nunique" not in agg.columns:
        # In rare multi-index misname, rebuild
        agg = df.groupby("Invoice").agg(
            ItemCount=("Quantity", "sum"),
            AvgUnitPrice=("UnitPrice", "mean"),
            MaxUnitPrice=("UnitPrice", "max"),
            MinUnitPrice=("UnitPrice", "min"),
            UniqueProducts=("StockCode", "nunique"),
            InvoiceValue=("TotalPrice", "sum"),
        )
        X = agg[["ItemCount", "AvgUnitPrice", "MaxUnitPrice", "MinUnitPrice", "UniqueProducts"]]
    y = agg["InvoiceValue"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    with open(out_dir / "regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance plot
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    importances = rf.feature_importances_
    ax.bar(range(len(importances)), importances)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(X.columns, rotation=30, ha="right")
    ax.set_title("RandomForest Feature Importances")
    ax.set_ylabel("Importance")
    plot_save(fig, out_dir / "feature_importance.png")

    # Save predictions
    preds = pd.DataFrame({"Invoice": X_test.index, "y_true": y_test, "y_pred": y_pred})
    preds.to_csv(out_dir / "predictions.csv", index=False)

    return metrics

def run_all(data_path: str, out_dir: str):
    out = ensure_out(out_dir)
    print(">> Loading data from:", data_path)
    df0 = load_data(data_path)
    print(">> Raw shape:", df0.shape)
    df = clean_data(df0)
    print(">> Clean shape:", df.shape)

    print(">> Running EDA...")
    basic_eda(df, out)

    print(">> RFM + KMeans segmentation...")
    rfm_info = rfm_kmeans(df, out)
    if rfm_info:
        print("   Best k:", rfm_info["best_k"], "Silhouette:", round(rfm_info["silhouette"], 3))
    else:
        print("   Skipped RFM (CustomerID not found)")

    print(">> Invoice value regression...")
    metrics = invoice_value_regression(df, out)
    print("   Metrics:", metrics)

    print(">> All done. Artifacts saved under:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to online_retail_II.xlsx or .csv")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()
    run_all(args.data_path, args.out_dir)
