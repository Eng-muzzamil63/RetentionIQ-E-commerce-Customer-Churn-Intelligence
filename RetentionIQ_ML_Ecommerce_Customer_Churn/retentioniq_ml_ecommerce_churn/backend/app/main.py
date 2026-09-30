from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ml import ChurnEngine, FEATURES

app = FastAPI(title="RetentionIQ API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = ChurnEngine()


class PredictionRequest(BaseModel):
    age: int = Field(35, ge=18, le=90)
    days_since_last_purchase: int = Field(45, ge=0)
    orders_90d: int = Field(2, ge=0)
    avg_order_value: float = Field(80, ge=0)
    support_tickets_90d: int = Field(1, ge=0)
    refunds_180d: int = Field(0, ge=0)
    discount_usage_rate: float = Field(.35, ge=0, le=1)
    session_days_30d: int = Field(7, ge=0)
    email_open_rate: float = Field(.45, ge=0, le=1)
    site_visits_30d: int = Field(10, ge=0)
    days_as_customer: int = Field(400, ge=1)
    lifetime_orders: int = Field(9, ge=1)
    lifetime_value: float = Field(850, ge=0)
    mobile_share: float = Field(.7, ge=0, le=1)
    acquisition_channel: str = "Organic"
    membership: str = "Standard"
    region: str = "Central"


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "RetentionIQ API", "status": "online", "docs": "/docs"}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "model": engine.selected_name}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    return engine.summary()


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"selected": engine.selected_name, "models": engine.model_metrics()}


@app.get("/api/customers")
def customers(
    limit: int = Query(80, ge=1, le=500),
    risk: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    rows = engine.customer_view(5000)
    if risk:
        rows = [r for r in rows if r["risk"].lower() == risk.lower()]
    if search:
        q = search.lower()
        rows = [r for r in rows if q in r["customer_id"].lower() or q in r["membership"].lower() or q in r["region"].lower()]
    return rows[:limit]


@app.get("/api/customers/{customer_id}")
def customer_detail(customer_id: str) -> dict[str, Any]:
    matches = engine.data[engine.data["customer_id"] == customer_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    row = matches.iloc[0]
    prediction = engine.predict_row(row)
    return {
        "profile": {k: row[k] for k in ["customer_id", "age", "membership", "region", "acquisition_channel", "days_as_customer", "lifetime_orders", "lifetime_value"]},
        "behavior": {k: row[k] for k in ["days_since_last_purchase", "orders_90d", "avg_order_value", "support_tickets_90d", "refunds_180d", "discount_usage_rate", "session_days_30d", "email_open_rate", "site_visits_30d", "mobile_share"]},
        "prediction": prediction,
        "drivers": engine.explain(row),
        "retention_action": engine.action_for_row(row, prediction["probability"]),
    }


@app.get("/api/segments")
def segments() -> list[dict[str, Any]]:
    return engine.segments()


@app.get("/api/retention-actions")
def retention_actions() -> list[dict[str, Any]]:
    rows = engine.customer_view(5000)
    actions: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        actions.setdefault(r["retention_action"], []).append(r)
    out = []
    for action, group in actions.items():
        out.append({
            "action": action,
            "customers": len(group),
            "estimated_value": round(sum(float(x["lifetime_value"]) for x in group), 2),
        })
    return sorted(out, key=lambda x: x["estimated_value"], reverse=True)


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    payload = request.model_dump()
    prediction = engine.predict_row(payload)
    return {
        **prediction,
        "drivers": engine.explain(payload),
        "retention_action": engine.action_for_row(payload, prediction["probability"]),
        "model": engine.selected_name,
    }
