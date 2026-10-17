# 💊 Medicine Stock-Out Prediction System for Defence

## Overview
A machine learning project that predicts the risk of medicine stock-outs using historical inventory and demand data to support timely replenishment in defence medical facilities.

## Features
- Stock-out risk prediction using Random Forest.
- Medicine inventory monitoring.
- Store and medicine selection.
- Interactive dashboard with Streamlit.
- Inventory visualization.

## Tech Stack
- Python, Pandas
- Scikit-learn
- Streamlit
- Joblib
- GitHub

## Dataset
- `dispensing.csv` – Inventory, demand, and stock-out records.
- `drugs.csv` – Medicine details.
- `stores.csv` – Store information.
- `context.csv` – Regional and seasonal data.

## Run Locally
```bash
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

## Future Scope
- Automated low-stock alerts.
- Demand forecasting.
- Reorder quantity recommendations.

## Note
This is an educational prototype. Predictions require further validation before real-world use. Use only non-sensitive data in public repositories.
