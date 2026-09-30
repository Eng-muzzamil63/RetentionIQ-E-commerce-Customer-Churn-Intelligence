# RetentionIQ — E-commerce Customer Churn Intelligence

A portfolio-ready ML product that predicts customer churn and turns model output into retention actions.

## What it demonstrates
- Binary classification with Logistic Regression and Random Forest
- Model comparison with ROC-AUC, precision, recall and F1
- Feature engineering from customer behavior
- Class-imbalance handling with class weights
- Customer-level churn probability and risk band
- SHAP-like local explanation using model coefficients / feature contributions (no heavy SHAP dependency)
- Customer segmentation and retention recommendations
- FastAPI API
- Next.js dashboard with interactive customer drill-down
- Fully synthetic demo dataset so the project runs without external data

## Folder structure
- `backend/` FastAPI + scikit-learn ML service
- `frontend/` Next.js + React dashboard

## Backend
```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/docs`.

## Frontend
In another terminal:
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000`.

## Demo credentials
No login is required.

## API endpoints
- `GET /api/health`
- `GET /api/summary`
- `GET /api/models`
- `GET /api/customers`
- `GET /api/customers/{customer_id}`
- `GET /api/segments`
- `GET /api/retention-actions`
- `POST /api/predict`

## Portfolio positioning
Instead of presenting this as “customer churn prediction,” position it as:

> **RetentionIQ — ML-powered customer churn intelligence for e-commerce teams.**

The key product flow is `customer behavior → ML probability → risk drivers → retention action`.
