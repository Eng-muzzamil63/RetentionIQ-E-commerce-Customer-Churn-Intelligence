from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

NUMERIC = [
    "age", "days_since_last_purchase", "orders_90d", "avg_order_value",
    "support_tickets_90d", "refunds_180d", "discount_usage_rate",
    "session_days_30d", "email_open_rate", "site_visits_30d",
    "days_as_customer", "lifetime_orders", "lifetime_value", "mobile_share",
]
CATEGORICAL = ["acquisition_channel", "membership", "region"]
FEATURES = NUMERIC + CATEGORICAL


def generate_demo_data(n: int = 5000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 66, n)
    days_since_last_purchase = np.clip(rng.gamma(2.5, 18, n).astype(int), 1, 240)
    orders_90d = np.clip(rng.poisson(2.8, n), 0, 15)
    avg_order_value = np.round(np.clip(rng.normal(82, 28, n), 18, 260), 2)
    support_tickets_90d = np.clip(rng.poisson(0.8, n), 0, 8)
    refunds_180d = np.clip(rng.poisson(0.55, n), 0, 6)
    discount_usage_rate = np.round(np.clip(rng.beta(2.2, 3.8, n), 0, 1), 3)
    session_days_30d = np.clip(rng.poisson(7.5, n), 0, 30)
    email_open_rate = np.round(np.clip(rng.beta(4, 2.5, n), 0, 1), 3)
    site_visits_30d = np.clip(rng.poisson(11, n), 0, 60)
    days_as_customer = np.clip(rng.gamma(3.2, 240, n).astype(int) + 30, 30, 2400)
    lifetime_orders = np.maximum(1, (days_as_customer / 45 + rng.normal(3, 4, n)).astype(int))
    lifetime_value = np.round(np.clip(lifetime_orders * avg_order_value * rng.uniform(0.8, 1.35, n), 30, 40000), 2)
    mobile_share = np.round(np.clip(rng.beta(4, 2, n), 0, 1), 3)

    acquisition_channel = rng.choice(["Organic", "Paid Search", "Social", "Email", "Referral"], n, p=[.28,.23,.18,.18,.13])
    membership = rng.choice(["Standard", "Plus", "VIP"], n, p=[.64,.27,.09])
    region = rng.choice(["North", "South", "East", "West", "Central"], n)

    # Latent churn propensity designed to create realistic but imperfect signal.
    score = (
        -2.35
        + 0.028 * days_since_last_purchase
        - 0.16 * orders_90d
        - 0.045 * session_days_30d
        - 0.018 * site_visits_30d
        - 0.85 * email_open_rate
        + 0.23 * support_tickets_90d
        + 0.30 * refunds_180d
        + 0.75 * discount_usage_rate
        - 0.00035 * days_as_customer
        - 0.18 * (membership == "Plus")
        - 0.50 * (membership == "VIP")
        + 0.12 * (acquisition_channel == "Paid Search")
        + 0.18 * (acquisition_channel == "Social")
        + rng.normal(0, 0.55, n)
    )
    probability = 1 / (1 + np.exp(-score))
    churn = (rng.random(n) < probability).astype(int)

    return pd.DataFrame({
        "customer_id": [f"CUS-{i:05d}" for i in range(1, n + 1)],
        "age": age,
        "days_since_last_purchase": days_since_last_purchase,
        "orders_90d": orders_90d,
        "avg_order_value": avg_order_value,
        "support_tickets_90d": support_tickets_90d,
        "refunds_180d": refunds_180d,
        "discount_usage_rate": discount_usage_rate,
        "session_days_30d": session_days_30d,
        "email_open_rate": email_open_rate,
        "site_visits_30d": site_visits_30d,
        "days_as_customer": days_as_customer,
        "lifetime_orders": lifetime_orders,
        "lifetime_value": lifetime_value,
        "mobile_share": mobile_share,
        "acquisition_channel": acquisition_channel,
        "membership": membership,
        "region": region,
        "churn": churn,
    })


@dataclass
class ModelBundle:
    pipeline: Pipeline
    name: str
    metrics: dict[str, float]


class ChurnEngine:
    def __init__(self) -> None:
        self.data = generate_demo_data()
        self.models: dict[str, ModelBundle] = {}
        self.selected_name = "Random Forest"
        self._train()

    def _preprocessor(self) -> ColumnTransformer:
        num = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        cat = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])

    def _train(self) -> None:
        X = self.data[FEATURES]
        y = self.data["churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
        )
        candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1600, class_weight="balanced", random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(
                n_estimators=360, max_depth=12, min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
            ),
        }
        for name, estimator in candidates.items():
            pipe = Pipeline([("pre", self._preprocessor()), ("model", estimator)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]
            metrics = {
                "roc_auc": float(roc_auc_score(y_test, proba)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_test, pred)),
            }
            self.models[name] = ModelBundle(pipe, name, metrics)

    @property
    def model(self) -> Pipeline:
        return self.models[self.selected_name].pipeline

    def risk_band(self, probability: float) -> str:
        if probability >= 0.75:
            return "Critical"
        if probability >= 0.50:
            return "High"
        if probability >= 0.25:
            return "Watch"
        return "Healthy"

    def predict_row(self, row: pd.Series | dict[str, Any]) -> dict[str, Any]:
        df = pd.DataFrame([dict(row)])[FEATURES]
        probability = float(self.model.predict_proba(df)[0, 1])
        return {"probability": probability, "risk": self.risk_band(probability)}

    def explain(self, row: pd.Series | dict[str, Any]) -> list[dict[str, Any]]:
        raw = pd.DataFrame([dict(row)])[FEATURES]
        model = self.model.named_steps["model"]
        pre = self.model.named_steps["pre"]
        encoded = pre.transform(raw)
        feature_names = list(pre.get_feature_names_out())
        if hasattr(model, "coef_"):
            contribution = encoded.toarray()[0] * model.coef_[0]
        else:
            # RF contribution proxy: normalized feature importance multiplied by centered input signal.
            dense = encoded.toarray()[0] if hasattr(encoded, "toarray") else encoded[0]
            importances = model.feature_importances_
            contribution = importances * np.tanh(dense)
        idx = np.argsort(np.abs(contribution))[::-1][:7]
        pretty = []
        for i in idx:
            name = feature_names[i].replace("num__", "").replace("cat__", "")
            pretty.append({
                "feature": name.replace("_", " ").title(),
                "impact": float(contribution[i]),
                "direction": "increases risk" if contribution[i] > 0 else "reduces risk",
            })
        return pretty

    def customer_view(self, limit: int = 100) -> list[dict[str, Any]]:
        subset = self.data.head(limit).copy()
        proba = self.model.predict_proba(subset[FEATURES])[:, 1]
        subset["churn_probability"] = proba
        subset["risk"] = [self.risk_band(float(p)) for p in proba]
        subset["retention_action"] = [self.action_for_row(r, float(p)) for (_, r), p in zip(subset.iterrows(), proba)]
        cols = ["customer_id", "churn_probability", "risk", "membership", "region", "orders_90d", "days_since_last_purchase", "lifetime_value", "retention_action"]
        return subset[cols].sort_values("churn_probability", ascending=False).to_dict("records")

    def action_for_row(self, row: pd.Series | dict[str, Any], probability: float | None = None) -> str:
        d = dict(row)
        p = probability if probability is not None else self.predict_row(d)["probability"]
        if p >= .75 and d.get("lifetime_value", 0) >= 800:
            return "VIP save offer"
        if p >= .75:
            return "Win-back sequence"
        if p >= .50:
            return "Personalized incentive"
        if d.get("orders_90d", 0) <= 1:
            return "Re-engagement journey"
        return "Nurture & monitor"

    def summary(self) -> dict[str, Any]:
        scored = self.customer_view(5000)
        df = pd.DataFrame(scored)
        counts = df["risk"].value_counts().to_dict()
        monthly = self.data.groupby(pd.cut(self.data["days_since_last_purchase"], bins=[0,15,30,60,90,180,300], include_lowest=True), observed=False)["churn"].mean().round(3)
        return {
            "customers": int(len(self.data)),
            "churn_rate": float(self.data["churn"].mean()),
            "high_risk_customers": int((df["churn_probability"] >= .50).sum()),
            "critical_customers": int((df["churn_probability"] >= .75).sum()),
            "retention_value_at_risk": float(df.loc[df["churn_probability"] >= .50, "lifetime_value"].sum()),
            "risk_bands": {k: int(v) for k, v in counts.items()},
            "recent_recency_curve": [{"bucket": str(k), "churn_rate": float(v)} for k, v in monthly.items()],
            "top_action": "VIP save offer",
        }

    def segments(self) -> list[dict[str, Any]]:
        out = []
        for membership, group in self.data.groupby("membership"):
            g = group.copy()
            p = self.model.predict_proba(g[FEATURES])[:, 1]
            out.append({
                "segment": membership,
                "customers": int(len(g)),
                "avg_churn_probability": float(np.mean(p)),
                "avg_ltv": float(g["lifetime_value"].mean()),
                "orders_90d": float(g["orders_90d"].mean()),
            })
        return sorted(out, key=lambda x: x["avg_churn_probability"], reverse=True)

    def model_metrics(self) -> list[dict[str, Any]]:
        return [{"name": name, **bundle.metrics} for name, bundle in self.models.items()]
