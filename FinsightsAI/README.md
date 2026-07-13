# FinSight AI — Financial Market Prediction System

> An AI-powered system combining **technical stock indicators** and **NLP sentiment analysis** to predict next-day stock price direction (Up ↑ / Down ↓).

---

## 📁 Project Structure

```
Final_Project/
├── data/
│   ├── raw/               # Raw CSVs (Yahoo Finance + Kaggle news)
│   ├── processed/         # Cleaned, feature-engineered datasets
│   └── models/            # Saved model .pkl files + SHAP plots
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_explainability.ipynb
├── src/
│   ├── utils.py            # Shared constants & logger
│   ├── data_loader.py      # Yahoo Finance + news CSV loader
│   ├── preprocessor.py     # OHLCV + text cleaning
│   ├── sentiment.py        # VADER pipeline + daily aggregation
│   ├── feature_engineer.py # Technical indicators + feature merging
│   ├── models.py           # Model training, evaluation, persistence
│   └── explainer.py        # SHAP analysis
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── run_pipeline.py         # End-to-end runner script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd Final_Project

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

### 2. Download Data

**Stock data** is fetched automatically via `yfinance`.

**News data** — Download from Kaggle manually:
1. Go to: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
2. Download `all-data.csv`
3. Place it in `data/raw/all-data.csv`

### 3. Run the Pipeline

```bash
python run_pipeline.py --news-path data/raw/all-data.csv
```

This will:
- Fetch 5 years of OHLCV data for AAPL, MSFT, NVDA, TSLA
- Preprocess and clean all data
- Run VADER sentiment analysis + daily aggregation
- Engineer 30+ features (technical + sentiment + temporal)
- Train 4 ML models with chronological split
- Save all models and metrics

### 4. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Features Engineered

### Technical Indicators
| Feature | Description |
|---|---|
| SMA 5/10/20 | Simple Moving Average |
| EMA 12/26 | Exponential Moving Average |
| RSI | Relative Strength Index (14-day) |
| MACD | 12/26/9 MACD + Signal + Histogram |
| Bollinger Bands | Upper/Lower/Width/%B (20-day, 2σ) |
| Rolling Volatility | 5-day & 10-day return std |

### Sentiment Features
| Feature | Description |
|---|---|
| avg_sentiment | Mean daily sentiment score (-1 to +1) |
| sentiment_volatility | Intraday sentiment standard deviation |
| pos_count | Count of positive headlines |
| neg_count | Count of negative headlines |
| neu_count | Count of neutral headlines |

### Temporal Features
- Day of week, month, is_month_end
- Lag-1, Lag-3, Lag-5 daily returns

---

## 🤖 Models

| Model | Library | Notes |
|---|---|---|
| Logistic Regression | scikit-learn | With StandardScaler |
| Random Forest | scikit-learn | 300 trees, balanced class weight |
| XGBoost | xgboost | 300 rounds, early stopping |
| LightGBM | lightgbm | 300 rounds, balanced class weight |

**Train/Val/Test split (chronological):**
- Train: 2019–2021
- Validation: 2022
- Test: 2023–2024

---

## 📈 Dashboard Pages

1. **📊 Market Overview** — Candlestick, MA overlay, Bollinger Bands, RSI, MACD
2. **💬 Sentiment Trends** — Daily sentiment score, weekly news counts, scatter vs returns
3. **🔮 Prediction Engine** — Date picker → UP/DOWN prediction + probability gauge
4. **🔍 Explainability** — SHAP global importance, waterfall plots, cross-model comparison
5. **📉 Model Performance** — Metrics table, ROC curves, confusion matrices

---

## ⚠️ Assumptions & Limitations

- **News-to-date alignment**: The Kaggle dataset has no timestamps. We use a synthetic rolling assignment for demonstration. In production, real news timestamps should be aligned with trading dates (after-hours news → next trading day).
- **Stationarity**: No explicit stationarity tests (e.g., ADF). Returns are used as proxies for stationary features.
- **Market regime**: Model trained on 2019–2021 may underperform during structural shifts (e.g., COVID crash, 2022 rate hikes).
- **No short-selling**: Trading signal accuracy assumes long-only positions.
- **Generic news**: The Kaggle dataset contains general financial headlines, not stock-specific. A production system should use stock-specific news APIs (e.g., NewsAPI, Benzinga).

---

## 📋 Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Precision | Avoids false "Buy" signals |
| Recall | Captures actual upward moves |
| F1 Score | Balance of precision & recall |
| ROC-AUC | Ranking quality of probability scores |

---

## 💡 Business Recommendations

1. **Investment Firms**: Use the XGBoost/LightGBM predictions as one signal in a multi-factor model; never as a standalone trading trigger.
2. **Hedge Funds**: Combine with order-book data and alternative data sources for alpha generation.
3. **Retail Investors**: Use the dashboard's sentiment trend page to understand market mood before making decisions.
4. **FinTech Platforms**: Integrate the prediction API endpoint into automated alert systems.

---

## 📦 Tech Stack

`Python` · `yfinance` · `scikit-learn` · `XGBoost` · `LightGBM` · `VADER` · `SHAP` · `Streamlit` · `Plotly` · `pandas` · `numpy`

---

## 👤 Author

Built as a Final Project for the AI-Powered Financial Market Prediction System course.
