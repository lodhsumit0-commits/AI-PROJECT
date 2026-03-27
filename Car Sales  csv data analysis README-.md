# 🚘 Car Sales Data Analysis
### Python · Pandas · Jupyter · SQL · Machine Learning · Power BI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-f7931e?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A complete **end-to-end data analysis and machine learning project** built on a car sales CSV dataset. This project covers data cleaning, exploratory data analysis (EDA), interactive visualizations, SQL querying, and a price prediction model — all from a single structured dataset.

---

## 📊 Project Overview

```
car_sales.csv
      │
      ▼
[Data Cleaning & Preprocessing]
      │
      ▼
[Exploratory Data Analysis (EDA)]
      │
      ├──► [Visualizations & Insights]
      │
      ├──► [SQL Queries & Aggregations]
      │
      └──► [ML Price Prediction Model]
                    │
                    ▼
           [Dashboard / Report]
```

---

## 🚀 Features

- ✅ Load, clean, and preprocess raw car sales CSV data
- ✅ Handle missing values, duplicates, and data type conversions
- ✅ Exploratory Data Analysis (EDA) with statistical summaries
- ✅ Rich visualizations — histograms, heatmaps, scatter plots, bar charts
- ✅ SQL-style queries using `pandas` and `SQLite`
- ✅ Machine learning model to predict car sale price
- ✅ Feature importance analysis
- ✅ Auto-generated HTML EDA report
- ✅ Power BI / Tableau-ready data export
- ✅ Fully documented Jupyter Notebooks

---

## 📁 Project Structure

```
car-sales-analysis/
│
├── data/
│   ├── car_sales.csv                   # Raw dataset
│   └── car_sales_cleaned.csv           # Cleaned dataset (generated)
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb          # Data loading & cleaning
│   ├── 02_eda.ipynb                    # Exploratory data analysis
│   ├── 03_visualizations.ipynb         # Charts and plots
│   ├── 04_sql_queries.ipynb            # SQL analysis with SQLite
│   └── 05_price_prediction.ipynb       # ML model for price prediction
│
├── src/
│   ├── clean.py                        # Data cleaning functions
│   ├── analyze.py                      # EDA helper functions
│   ├── model.py                        # ML model training & evaluation
│   └── utils.py                        # Shared utility functions
│
├── output/
│   ├── figures/                        # Saved plots (PNG)
│   └── car_sales_report.html           # Auto-generated EDA report
│
├── dashboard/
│   └── car_sales_dashboard.pbix        # Power BI dashboard file
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📋 Dataset Description

The dataset `car_sales.csv` contains individual vehicle transaction records with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `Car_id` | int | Unique record identifier |
| `Date` | datetime | Date of the sale |
| `Customer_Name` | string | Buyer's full name |
| `Gender` | string | Buyer's gender |
| `Annual_Income` | float | Buyer's annual income (USD) |
| `Dealer_Name` | string | Name of the dealership |
| `Company` | string | Car manufacturer / brand |
| `Model` | string | Car model name |
| `Engine` | string | Engine type (e.g., Double Overhead Camshaft) |
| `Transmission` | string | Automatic or Manual |
| `Color` | string | Exterior color |
| `Price ($)` | float | Final sale price in USD |
| `Dealer_No` | string | Dealer identification number |
| `Body_Style` | string | SUV, Sedan, Hatchback, Coupe, etc. |
| `Phone` | string | Buyer's contact number |
| `Dealer_Region` | string | Geographic sales region |

> 📌 Column names may vary slightly depending on your CSV version. Adjust in `src/utils.py` as needed.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/car-sales-analysis.git
cd car-sales-analysis
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.3.0
jupyter>=1.0.0
sqlalchemy>=2.0.0
ydata-profiling>=4.0.0
openpyxl>=3.1.0
```

### 4. Launch Jupyter Notebooks

```bash
jupyter notebook
```

---

## 🖥️ Usage

### Quick Start — Full Pipeline

```bash
python src/clean.py   --input  data/car_sales.csv \
                      --output data/car_sales_cleaned.csv

python src/analyze.py --input  data/car_sales_cleaned.csv

python src/model.py   --input  data/car_sales_cleaned.csv
```

### Run Notebooks in Order

```
01_data_cleaning.ipynb
      ↓
02_eda.ipynb
      ↓
03_visualizations.ipynb
      ↓
04_sql_queries.ipynb
      ↓
05_price_prediction.ipynb
```

---

## 🧹 Data Cleaning

```python
import pandas as pd

df = pd.read_csv("data/car_sales.csv")

# 1. Drop duplicates
df.drop_duplicates(inplace=True)

# 2. Handle missing values
df["Annual_Income"].fillna(df["Annual_Income"].median(), inplace=True)
df.dropna(subset=["Price ($)", "Company", "Model"], inplace=True)

# 3. Fix data types
df["Date"]     = pd.to_datetime(df["Date"])
df["Price ($)"] = pd.to_numeric(df["Price ($)"], errors="coerce")

# 4. Standardize text columns
df["Gender"]       = df["Gender"].str.strip().str.title()
df["Transmission"] = df["Transmission"].str.strip().str.title()

# 5. Extract time features
df["Year"]    = df["Date"].dt.year
df["Month"]   = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter

df.to_csv("data/car_sales_cleaned.csv", index=False)
print(f"✅ Cleaned: {df.shape[0]:,} rows × {df.shape[1]} columns")
```

---

## 📊 Exploratory Data Analysis

### Key Questions Answered

- 📌 Which car brands generate the most total revenue?
- 📌 What is the distribution of sale prices across the dataset?
- 📌 How do monthly and quarterly sales trends look?
- 📌 Does buyer income correlate with purchase price?
- 📌 Which body styles and transmission types sell the most?
- 📌 What are the top-performing dealerships and regions?

### Summary Statistics

```python
print(df.describe())
print(f"\nUnique brands   : {df['Company'].nunique()}")
print(f"Unique models   : {df['Model'].nunique()}")
print(f"Date range      : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Avg sale price  : ${df['Price ($)'].mean():,.0f}")
```

---

## 📈 Visualizations

### Top Brands by Revenue

```python
import matplotlib.pyplot as plt
import seaborn as sns

top_brands = (df.groupby("Company")["Price ($)"]
                .sum()
                .sort_values(ascending=False)
                .head(10))

plt.figure(figsize=(12, 5))
sns.barplot(x=top_brands.index, y=top_brands.values, palette="viridis")
plt.title("Top 10 Car Brands by Total Sales Revenue")
plt.xlabel("Brand")
plt.ylabel("Total Revenue ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/figures/top_brands_revenue.png")
plt.show()
```

### Price Distribution

```python
plt.figure(figsize=(10, 4))
sns.histplot(df["Price ($)"], bins=50, kde=True, color="steelblue")
plt.title("Car Sale Price Distribution")
plt.xlabel("Price ($)")
plt.tight_layout()
plt.show()
```

### Monthly Sales Trend

```python
monthly = (df.groupby(["Year", "Month"])["Price ($)"]
             .sum()
             .reset_index())
monthly["Period"] = pd.to_datetime(
    monthly[["Year", "Month"]].assign(Day=1)
)

plt.figure(figsize=(14, 4))
plt.plot(monthly["Period"], monthly["Price ($)"],
         marker="o", color="darkorange")
plt.title("Monthly Sales Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue ($)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Correlation Heatmap

```python
plt.figure(figsize=(6, 4))
sns.heatmap(df[["Price ($)", "Annual_Income"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
```

---

## 🗄️ SQL Analysis

Query the dataset with SQL using SQLite in-memory:

```python
import sqlite3, pandas as pd

df = pd.read_csv("data/car_sales_cleaned.csv")
conn = sqlite3.connect(":memory:")
df.to_sql("car_sales", conn, index=False, if_exists="replace")

# ── Top 5 dealers by total revenue ──────────────────────────────────
q1 = """
    SELECT  Dealer_Name,
            COUNT(*)                    AS total_sales,
            ROUND(SUM([Price ($)]), 2)  AS total_revenue,
            ROUND(AVG([Price ($)]), 2)  AS avg_price
    FROM    car_sales
    GROUP   BY Dealer_Name
    ORDER   BY total_revenue DESC
    LIMIT   5;
"""
print(pd.read_sql_query(q1, conn))

# ── Sales by region and body style ──────────────────────────────────
q2 = """
    SELECT  Dealer_Region,
            Body_Style,
            COUNT(*) AS units_sold
    FROM    car_sales
    GROUP   BY Dealer_Region, Body_Style
    ORDER   BY units_sold DESC;
"""
print(pd.read_sql_query(q2, conn))

# ── Average price by transmission type ──────────────────────────────
q3 = """
    SELECT  Transmission,
            ROUND(AVG([Price ($)]), 2) AS avg_price,
            COUNT(*)                   AS count
    FROM    car_sales
    GROUP   BY Transmission;
"""
print(pd.read_sql_query(q3, conn))

conn.close()
```

---

## 🤖 Price Prediction Model

Train a **Random Forest Regressor** to predict sale price from buyer and vehicle features.

```python
from sklearn.model_selection    import train_test_split
from sklearn.ensemble           import RandomForestRegressor
from sklearn.preprocessing      import LabelEncoder
from sklearn.metrics            import mean_absolute_error, r2_score
import numpy as np

FEATURES = ["Annual_Income", "Company", "Transmission",
            "Engine", "Body_Style", "Gender",
            "Dealer_Region", "Month", "Quarter"]
TARGET   = "Price ($)"

df_model = df[FEATURES + [TARGET]].dropna().copy()

# Encode categorical columns
le = LabelEncoder()
for col in df_model.select_dtypes(include="object").columns:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"MAE   : ${mean_absolute_error(y_test, y_pred):>10,.2f}")
print(f"RMSE  : ${np.sqrt(np.mean((y_test - y_pred)**2)):>10,.2f}")
print(f"R²    :  {r2_score(y_test, y_pred):.4f}")
```

### Feature Importance

```python
importances = (pd.Series(model.feature_importances_, index=FEATURES)
                 .sort_values(ascending=False))

plt.figure(figsize=(10, 5))
importances.plot(kind="bar", color="teal")
plt.title("Feature Importance — Car Price Prediction")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("output/figures/feature_importance.png")
plt.show()
```

---

## 🗺️ Auto EDA Report

Generate a full interactive profiling report in one line:

```python
from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("data/car_sales_cleaned.csv")
profile = ProfileReport(df, title="Car Sales EDA Report", explorative=True)
profile.to_file("output/car_sales_report.html")
print("✅ Report saved → output/car_sales_report.html")
```

Open the generated file in any browser for a complete interactive report including distributions, correlations, missing values, and duplicate analysis.

---

## 📊 Sample Insights

| Metric | Value |
|--------|-------|
| Total records | ~23,000+ |
| Unique car brands | 15+ |
| Price range | $1,000 – $85,000+ |
| Top brand by revenue | Chevrolet / Ford |
| Best-selling body style | SUV |
| Most common transmission | Automatic |
| Peak sales month | December |

> ⚠️ Values above are illustrative — actual results depend on your dataset.

---

## 📌 Roadmap

- [ ] Add time-series sales forecasting (Prophet / ARIMA)
- [ ] Build interactive Streamlit web app
- [ ] Add customer segmentation via clustering (K-Means)
- [ ] Deploy price prediction model as a REST API (FastAPI)
- [ ] Integrate Power BI auto-refresh pipeline
- [ ] Add unit tests for cleaning and modeling scripts

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Kaggle — Car Sales Dataset](https://www.kaggle.com/) — Original data source
- [Pandas](https://pandas.pydata.org/) — Data manipulation
- [Scikit-Learn](https://scikit-learn.org/) — Machine learning
- [YData Profiling](https://github.com/ydataai/ydata-profiling) — Auto EDA reports
- [Plotly](https://plotly.com/python/) — Interactive visualizations

---

## 📬 Contact

**Your Name** — [@your-twitter](https://twitter.com/your-twitter) — your.email@example.com

Project Link: [https://github.com/your-username/car-sales-analysis](https://github.com/your-username/car-sales-analysis)
