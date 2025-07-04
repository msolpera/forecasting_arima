# Sales Forecasting 

This project focuses on predicting future sales using historical transaction data from a retail superstore. The dataset includes order-level information such as order date, product category, region, profit, and quantity sold. The goal is to build a machine learning pipeline to forecast sales across different segments and time periods.

---
---

## Dataset

The dataset used is based on the classic "Superstore" sample provided in Excel format (`Sample - Superstore.xls`).  
It contains sales records with the following key fields:

- `Order Date`: Date of purchase
- `Region`: Geographical region of the customer
- `Category` and `Sub-Category`: Product classification
- `Sales`: Target variable — total sales amount
- `Quantity`, `Discount`, `Profit`: Additional features

---

## Objective of this project

Build a robust forecasting model to predict future `Sales` over time, taking into account regional and category-level trends.  
This can support inventory planning, marketing, and business strategy.

---

## Approach

The project follows a typical data science pipeline:

1. **Data Cleaning & Exploration**
   - Handling missing values, formatting dates
   - Visualizing sales trends over time and across segments

2. **Feature Engineering**
   - Time-based features: month, year, lag variables
   - Rolling averages, promotion flags (if applicable)

3. **Modeling Approaches**

    - **Option A**: Univariate Forecasting (ARIMA):
    Create a separate time series for each product category by aggregating daily sales. An ARIMA model will be trained independently for each category to evaluate its ability to capture individual sales dynamics.

    - **Option B**: Multivariate Forecasting (XGBoost):
    In a second phase, we will consolidate the entire dataset into a unified table where each row represents a daily record per category. Using features such as lagged sales and time-based variables (e.g., month, day of week), we will train a global model with XGBoost to capture cross-category patterns and temporal dependencies.

    This dual approach allows us to compare traditional time series methods with modern machine learning models in terms of accuracy, scalability, and practical application.

    - Time series split for cross-validation

4. **Evaluation**
   - Metrics: RMSE, MAPE
   - Visualization of predicted vs. actual sales

---

## Tools & Libraries

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` for preprocessing and metrics
- `XGBoost`, `LightGBM`, or `CatBoost` for modeling
- `Prophet` or `statsmodels` (optional, for classical models)

---

## Forecasting Strategy

- Aggregate data by `Order Date`, `Region`, and `Category` as needed
- Create lag features and time-based rolling averages
- Tune and evaluate forecasting models across different horizons (e.g., next month, quarter)

---

## Visualizations

- Sales trends per category and region
- Seasonal patterns
- Forecasted vs. actual sales

---

## File Structure

project/
│
├── data/
│ └── Superstore.xls
│
├── notebooks/
│ └── 01_data_exploration.ipynb
│ └── 02_modeling.ipynb
│
├── src/
│ └── preprocessing.py
│ └── forecasting.py
│
├── README.md
└── requirements.txt