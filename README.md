# Brent Oil Price Prediction Backend

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2A9D8F?style=for-the-badge&logo=xgboost&logoColor=white)
![Turso](https://img.shields.io/badge/Turso-000000?style=for-the-badge&logo=turso&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)

> FastAPI backend for Brent crude oil price forecasting using a VMD-based ensemble model and FinBERT sentiment analysis.

---

## 📖 About This Project

This project provides a robust forecasting engine for Brent oil prices. It combines statistical (ARIMA), deep learning (GRU), and tree-based (XGBoost) models into a meta-ensemble to predict future returns, which are then converted into a 14-day price forecast. The system also integrates a sentiment analysis pipeline using FinBERT to ingest and process oil-related news, further refining the predictions based on market sentiment.

Key highlights:
- **Ensemble Pipeline**: Multi-model approach combining GRUs, ARIMA, and XGBoost.
- **Sentiment Integration**: Real-time news scraping and sentiment scoring via FinBERT.
- **Monitoring**: Built-in instrumentation for Prometheus and Grafana.
- **Explainability**: Integrated SHAP and model attribution diagnostics.

---

## ✨ Features

- 🚀 **14-day price forecast** - Generates a daily-locked forecast for the next two weeks.
- 📊 **Ensemble Model** - Combines components using a Ridge meta-ensemble for superior accuracy.
- 🧠 **Sentiment Analysis** - Leverages FinBERT to quantify the impact of oil-related news.
- 🔍 **Explainability** - Provides feature attributions and model diagnostics via the `/explain` endpoint.
- 📈 **Fan Chart Visualization** - Generates quantile bands for uncertainty visualization.
- 🛠️ **Automated Scraper** - Periodically fetches prices and news articles to keep data fresh.
- 🛡️ **Comprehensive Testing** - Over 200+ test cases covering API, services, and models.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | [FastAPI v0.104.0+](https://fastapi.tiangolo.com/) |
| ML Framework | [PyTorch v2.0.0+](https://pytorch.org/) |
| Boosted Trees | [XGBoost v2.0.0+](https://xgboost.readthedocs.io/) |
| Language | [Python v3.11.7](https://www.python.org/) |
| Database | [Turso (libsql)](https://turso.tech/) |
| Monitoring | [Prometheus](https://prometheus.io/) & [Grafana](https://grafana.com/) |
| Deployment | [Docker](https://www.docker.com/) |
| Sentiment | [FinBERT (via Transformers)](https://huggingface.co/ProsusAI/finbert) |

---

## 📋 Prerequisites

- [Python](https://www.python.org/) **v3.11.7 or higher**
- [pip](https://pip.pypa.io/)
- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/) (Optional, for containerized deployment)

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PramudithaN/fyp_backend.git
cd fyp_backend
```

### 2. Install dependencies

```bash
make setup
# OR
pip install -r requirements.txt -r requirements-dev.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
NEWSAPI_KEY=your_newsapi_key_here
NEWSDATA_KEY=your_newsdata_key_here
TURSO_DATABASE_URL=your_turso_url
TURSO_AUTH_TOKEN=your_turso_token
SENTIMENT_MODE=finbert
```

### 4. Start the development server

```bash
make run
# OR
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to view the API documentation.

---

## 📦 Available Scripts

| Command | Description |
|---------|-------------|
| `make install` | Install production dependencies |
| `make test` | Run all tests |
| `make test-cov` | Run tests with coverage report |
| `make lint` | Run code quality checks (flake8, mypy) |
| `make format` | Format code with black and isort |
| `make run` | Start the FastAPI server locally |
| `make docker-build` | Build the Docker container |

---

## 📁 Project Structure

```
fyp_backend/
├── app/                       # Main application source code
│   ├── models/                # ML model definitions (GRU, etc.)
│   ├── services/              # Business logic and orchestration
│   ├── schemas/               # Pydantic data models
│   └── main.py                # FastAPI entry point
├── model_artifacts/           # Trained model files (.pt, .pkl, .json)
├── scripts/                   # Data migration and maintenance scripts
├── tests/                     # Comprehensive test suite
├── grafana/                   # Grafana dashboard configurations
├── prometheus/                # Prometheus metrics configuration
├── Makefile                   # Project automation commands
└── requirements.txt           # Production dependencies
```

---

## 🧪 Testing & Quality

This project maintains high standards with over **200+ test cases** and automated analysis.

[![Tests](https://github.com/PramudithaN/fyp_backend/actions/workflows/tests.yml/badge.svg)](https://github.com/PramudithaN/fyp_backend/actions/workflows/tests.yml)
[![SonarCloud](https://github.com/PramudithaN/fyp_backend/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/PramudithaN/fyp_backend/actions/workflows/sonarcloud.yml)

Run tests locally:
```bash
make test-cov
```

---

## 🙋‍♂️ Connect with Me

- **GitHub**: [github.com/PramudithaN](https://github.com/PramudithaN)
- **LinkedIn**: [linkedin.com/in/pramuditha-nadun-612b1b204](https://linkedin.com/in/pramuditha-nadun-612b1b204)
- **Email**: pramudithanadun@gmail.com

---

*Developed with ❤️ by Pramuditha Nadun.*
