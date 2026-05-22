# 📦 Fan Sales Demand Forecasting — Inter Hall Data Analytics 2021-22

> **Competition:** Inter Hall Data Analytics 2021-22, hosted by Nihilent Analytics  
> **Team:** Team 278  
> **Task:** Forecast monthly fan sales by SKU by warehouse for June 2021

---

## 🏭 Problem Statement

XYZ company manufactures ceiling fans and table fans and distributes them across India through four regional warehouses (North, East, South, West). The warehouse manager must forecast dealer demand **one month in advance** to maintain inventory levels.

The company was facing high inventory buildup and poor fill rates due to inaccurate forecasting. The goal was to improve forecast accuracy using time series and machine learning methods.

**Evaluation Metric:** Mean Absolute Percentage Error (MAPE)

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Time Range | April 2018 – May 2021 (**38 months**) |
| Granularity | Monthly sales by SKU by Warehouse |
| SKUs | 408 unique SKUs |
| Warehouses | 4 (Wh-1 through Wh-4) |
| Regions | North, East, South, West |
| Rows | 1,039 |
| Target | Predict June 2021 sales for each SKU-Warehouse pair |

---

## 📁 Repository Structure

```
fan-sales-forecasting/
├── data/
│   ├── train_data.xlsx        # Historical sales data (Apr 2018 – May 2021)
│   ├── submission_format.xlsx # Expected output format
│   └── submission.csv         # Final predictions for June 2021
├── notebooks/
│   └── analysis.ipynb         # Full EDA, model experiments, and final predictions
├── reports/
│   ├── problem_statement.pdf  # Official competition problem statement
│   └── team_report.pdf        # Team 278's detailed approach report
├── src/
│   └── utils.py               # Helper functions (MAPE calculation, preprocessing)
└── README.md
```

---

## 🔍 Exploratory Data Analysis

Key observations from the data:

- **COVID-19 impact:** Clear dips in sales during April 2020 (first lockdown) and April–May 2021 (second wave) across all regions.
- **Strong month-to-month correlation:** Consecutive months show high positive correlation — the correlation heatmap confirmed this up to 2 months lag.
- **Non-stationarity:** ADF test revealed ~20% of SKU series are highly volatile; ~25–30% are near-constant (stationary).
- **Seasonality:** Peak demand observed in summer months (April–June) across most regions.

---

## 🤖 Models Explored

### Statistical Models
| Model | Description |
|---|---|
| **ARIMA** | Autoregressive Integrated Moving Average — uses lagged values and moving average errors |
| **SARIMA** | Seasonal ARIMA — extends ARIMA with seasonal differencing parameters (P, D, Q, m) |
| **SES** | Simple Exponential Smoothing — weighted average of past observations with smoothing factor α |
| **Holt Linear Trend** | Double exponential smoothing — models both level and trend components |

### Machine Learning Models
| Model | Description |
|---|---|
| **XGBoost Regressor** | Gradient boosting ensemble; strong baseline for tabular regression |
| **Random Forest** | Ensemble of decision trees using bagging |

### Deep Learning Models
| Model | Description |
|---|---|
| **LSTM** | Long Short-Term Memory network — handles sequential/temporal dependencies without vanishing gradient issues |

### Probabilistic Forecasting
| Model | Description |
|---|---|
| **FB Prophet** | Additive model capturing trend, seasonality, and holiday effects |

---

## ✅ Final Approach

After evaluating all models on a held-out validation set, we found that a **simple heuristic outperformed all complex models**:

```
Predicted June 2021 Sales = min(April 2021 Sales, May 2021 Sales)
```

**Why this works:**
1. Strong correlation between consecutive months makes the most recent months the best predictors.
2. The MAPE metric is asymmetric — under-forecasting is penalized less than over-forecasting (percentage error is bounded at 100% below, but unbounded above).
3. Taking the minimum of the two most recent months intentionally biases toward under-forecasting to exploit this asymmetry.

**Validation MAPE (May 2021):** `46.55%`

---

## 📈 Key Visualizations

- Total monthly sales across all warehouses (showing COVID dips)
- Region-wise rolling rate of change
- Quarterly region-wise sales breakdown (North, East, South, West)
- Correlation heatmap of same-month sales across years
- Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots
- ADF stationarity test scores across all SKU series
- Rolling mean and rolling standard deviation
- Geo-visualization of total sales by region and financial year

---

## 🧠 Insights

- All four regions showed pronounced COVID-19 impact in identical months, suggesting external events dominate over regional differences during disruptions.
- Correlation between June sales and the same month in prior years (June 2018, June 2019, June 2020) was weaker than expected, making year-over-year approaches less effective.
- Simple baselines grounded in recent data beat complex models, likely due to the short time series length (38 months) and high SKU-level variance.

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/<your-username>/fan-sales-forecasting.git
cd fan-sales-forecasting

# Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels prophet xgboost keras tensorflow openpyxl

# Open the notebook
jupyter notebook notebooks/analysis.ipynb
```

---

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FB Prophet Documentation](https://facebook.github.io/prophet/)
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [LSTM — Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
