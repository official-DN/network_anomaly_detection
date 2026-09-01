# AI-Based Anomaly Detection in Network Traffic

A complete production-style AI/ML platform for detecting abnormal network behavior and potential cyberattacks in real time.

## Overview

This system combines **supervised** (Random Forest, XGBoost) and **unsupervised** (Isolation Forest, Autoencoder) machine learning models to analyze network flows, detect anomalies, generate alerts, and display results on a cybersecurity dashboard.

## Architecture

```
Network Traffic (PCAP / CSV / Live)
         ↓
  Feature Extraction
         ↓
  Data Preprocessing (fitted on train only)
         ↓
  ┌──────────────┬─────────────────┐
  │ Supervised   │  Unsupervised   │
  │ Random Forest│  Isolation Forest│
  │ XGBoost      │  Autoencoder    │
  └──────┬───────┴────────┬────────┘
         └────────┬────────┘
              Detection Engine
                  ↓
          Risk Score (0-100)
                  ↓
         Alert Generation
                  ↓
      FastAPI Backend ←→ MongoDB
                  ↓
         React Dashboard
```

## Technology Stack

| Layer       | Technology                              |
|-------------|----------------------------------------|
| Language    | Python 3.11+                           |
| ML Models   | Random Forest, XGBoost, Isolation Forest, Autoencoder |
| Backend     | FastAPI + Uvicorn                      |
| Database    | MongoDB (Motor async driver)           |
| Frontend    | React + Recharts + Tailwind CSS        |
| Network     | Scapy (PCAP parsing, passive capture)  |
| Auth        | JWT (python-jose) + bcrypt             |

## Project Structure

```
network_anomaly_detection/
├── backend/          # FastAPI application
│   ├── api/          # Route handlers
│   ├── services/     # Business logic
│   ├── schemas/      # Pydantic models
│   └── utils/        # Logger, security
├── ml/               # Machine learning pipeline
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── train_*.py
├── network/          # Packet capture & parsing
├── frontend/         # React dashboard
│   └── src/
│       ├── components/
│       └── pages/
├── tests/            # pytest test suite
├── scripts/          # Utility scripts
└── models/           # Saved model files
```

## Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB (optional — falls back to in-memory)

### Backend Setup

```bash
# Clone repository
git clone <repo-url>
cd network_anomaly_detection

# Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Edit .env with your settings (MongoDB URL, secret key, etc.)
```

### Train Models

```bash
# Train all models with synthetic data (no dataset needed)
python scripts/train_all_models.py

# Or with a real dataset (e.g., CICIDS2017 CSV)
python scripts/train_all_models.py --data data/raw/dataset.csv
```

### Run Backend

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/api/docs

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Dashboard: http://localhost:5173

Default credentials: **admin / admin123**

## Dataset Support

Supported public cybersecurity datasets:
- CICIDS2017 / CSE-CIC-IDS2018
- UNSW-NB15
- NSL-KDD
- CIC-DDoS2019
- Bot-IoT / TON_IoT

The system uses automatic column mapping — no manual column renaming needed.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | /api/health | Health check |
| POST | /api/auth/login | Authenticate |
| POST | /api/traffic/upload-csv | Analyze CSV dataset |
| POST | /api/traffic/upload-pcap | Analyze PCAP file |
| GET  | /api/traffic/recent | Recent events |
| GET  | /api/traffic/statistics | Aggregate stats |
| POST | /api/detection/predict | Single prediction |
| POST | /api/detection/batch | Batch prediction |
| GET  | /api/alerts | List alerts |
| PATCH| /api/alerts/{id} | Update alert status |
| GET  | /api/models | Model info |
| GET  | /api/models/performance | Model metrics |
| GET  | /api/analytics/timeline | Traffic timeline |

## Generate Demo Traffic

```bash
# Submit synthetic flows to the running API
python scripts/generate_demo_traffic.py --n 100
```

## Run Tests

```bash
pytest tests/ -v
```

## Docker Deployment

```bash
docker-compose up --build
```

Services: backend (8000), frontend (5173), MongoDB (27017)

## Security Notice

This is a **defensive monitoring** system. All packet capture requires:
- Administrative / root privileges
- Explicit written authorization from the network owner
- Operation only on authorized networks or laboratory environments

## Risk Scoring

| Score | Level    | Action Required |
|-------|----------|-----------------|
| 0–29  | LOW      | Log & monitor |
| 30–59 | MEDIUM   | Review within 4h |
| 60–79 | HIGH     | Investigate promptly |
| 80–100| CRITICAL | Immediate response |

## User Roles

| Role    | Permissions |
|---------|-------------|
| ADMIN   | Full access |
| ANALYST | View + update alerts |
| VIEWER  | Read-only |

## Limitations

- Anomaly scores are model outputs, not definitive severity ratings
- Performance metrics require trained models
- Live capture requires root privileges and Scapy
- PCAP analysis requires Scapy installation

## Future Enhancements

- LSTM sequence-based anomaly detection
- Real-time WebSocket streaming
- SHAP explainability integration
- Automated model retraining pipeline
- Kibana / Grafana integration
