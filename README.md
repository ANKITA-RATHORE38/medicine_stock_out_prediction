# 💊 Medicine Stock-Out Prediction System for Defence

## 🚀 Live Demo

**Try the deployed application here:**

[Medicine Stock-Out Prediction System](https://medicinestockoutprediction-jbqfuy3kmydkvknwd3r6my.streamlit.app/)

## Overview

A machine learning project that predicts the risk of medicine stock-outs using historical inventory and demand data. It helps support timely replenishment and better inventory management in defence medical facilities.

## Features

- Stock-out risk prediction using Random Forest.
- Medicine inventory monitoring.
- Store and medicine selection.
- Interactive dashboard built with Streamlit.
- Inventory data visualization.

## Tech Stack

- **Programming Language:** Python
- **Data Processing:** Pandas
- **Machine Learning:** Scikit-learn
- **Web Dashboard:** Streamlit
- **Model Serialization:** Joblib
- **Version Control:** GitHub

## Dataset

The project uses the following datasets:

- `dispensing.csv` – Inventory, demand, and stock-out records.
- `drugs.csv` – Medicine details.
- `stores.csv` – Store information.
- `context.csv` – Regional and seasonal data.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ANKITA-RATHORE38/medicine_stock_out_prediction.git
cd medicine_stock_out_prediction
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train the machine learning model

```bash
python train_model.py
```

### 4. Launch the application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

## Future Scope

- Automated low-stock alerts.
- Medicine demand forecasting.
- Reorder quantity recommendations.
- Improved prediction accuracy using additional historical data.


