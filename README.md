# Retail Supply Chain & Demand Forecasting (`CAP3_Retail`)

## Executive Summary
Accurate demand forecasting is critical for optimizing retail supply chain operations—directly minimizing costly stockouts and bloated inventory holding costs. Built around **CRISP-DM** (domain logic and supply chain decision-making) and **Microsoft TDSP** (modular notebooks, persisted artifacts, and reproducible workflows), this capstone delivers an end-to-end machine learning and deep learning forecasting pipeline across **1,115 retail stores** spanning roughly 2.5 years of daily store-level data.

## Business Problem & Supply Chain Context

Accurate store-day demand forecasting underpins replenishment, safety stock planning, and promotional readiness across the store network:
- **Closed-Store Hygiene (`Open == 0`):** Enforcing `Sales = 0` and `Customers = 0` on inactive days prevents models from learning artificial demand noise or incurring false errors on closed storefronts.

- **Error Metric Choice (`RMSE` vs. `MAE`):** While **MAE** provides intuitive average unit error, **RMSE** penalizes large forecast misses more heavily—aligning with supply chain risk where severe stockouts or overstocks carry disproportionate operational costs.

- **Store Segmentation (`K-Means`):** Grouping stores by average daily **Sales** and **Customers** distinguishes high-volume flagship locations from mid-tier and lower-volume outlets for tailored inventory policies.

## Repository Structure (TDSP)
```text
CAP3_Retail/
├── data/
│   ├── raw/                  # train_data.csv, test_data.csv
│   └── processed/            # train_processed.csv, test_processed.csv, gbr_submission.csv
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_evaluation.ipynb
├── model/                    # gbr_retail_demand_model.pkl, scaler.pkl
├── outputs/
│   └── tables/               # model_performance_comparison.csv
└── src/                      # Modular scripts / utilities
```
## Pipeline Overview

- **Notebook 1 (01_eda_preprocessing.ipynb)**: Data ingestion, schema and missing-value audits, closed-store domain cleaning (Open == 0), date parsing, temporal feature engineering (Year, Month, Day, WeekOfYear), one-hot encoding (StateHoliday), skewness and IQR outlier diagnostics, StandardScaler normalization, chronological 80/20 splitting, and anchor export to data/processed/.

- **Notebook 2 (02_model_training.ipynb)**: Global Linear Regression baseline, HistGradientBoostingRegressor, store-specific linear models and weighted ensembling, RidgeCV/LassoCV, PCA + RandomizedSearchCV Random Forest, univariate time-series and seasonality profiling (highlighting November/December holiday peaks), permutation feature importance, and model/scaler serialization (joblib).

- **Notebook 3 (03_inference_evaluation.ipynb)**: Production artifact loading and test-set inference (gbr_submission.csv), LSTM sequence modeling on a 30-day sliding window, K-Means store clustering (Elbow and Silhouette evaluation, cluster-specific models), and feedforward ANN tuning (layer/neuron/batch/epoch experiments, Adam/RMSprop/SGD, ReLU/ELU, Dropout, and Random Search, Grid Search, and 3-Fold Cross-Validation).

## Key Takeaways & Results

- **Global Tree Ensembles Lead on Tabular Store Data**: HistGradientBoostingRegressor (Val RMSE: 1,131.01, Val MAE: 766.75) outperformed linear/regularized models (RMSE: ~1,498.59–1,502.91) and feedforward ANNs (RMSE: ~1,395.26–1,417.10) by capturing non-linear footfall and promotional thresholds across all stores.

- **Data Sparsity vs. Parameter Sharing**: Isolated per-store linear models suffered from high variance and sparse store histories (Avg RMSE: 6,460.57), whereas global parameter sharing and K-Means store segmentation (K = 3) balanced local store-tier behavior with sample density (achieving RMSE: 995.70 on Cluster 0).

- **Seasonality Drives Demand**: Network sales peak strongly in December across the vast majority of stores (1,008 of 1,115 stores), making calendar and promotional covariates essential for seasonal supply chain planning.

- **Primary Demand Drivers**: Primary demand drivers validate that customer foot traffic and store identity dominate sales predictions.


