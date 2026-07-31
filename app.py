from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Notebook → Production", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --ink:#0b1220; --blue:#4f7cff; --cyan:#6fe7dd; --paper:#f6f7fb; --muted:#76819a; --line:#dfe4ee; }
.stApp { background:var(--paper); color:var(--ink); font-family:'Space Grotesk',sans-serif; }
[data-testid='stSidebar'] { background:#0b1220; }
[data-testid='stSidebar'] * { color:#eef3ff !important; }
.hero { background:linear-gradient(135deg,#0b1220 0%,#182c57 65%,#2c4a89 100%); padding:3.4rem 3.6rem 3.2rem; border-radius:22px; color:#fff; margin:0 0 1.6rem; position:relative; overflow:hidden; }
.hero:after { content:'MLOPS'; position:absolute; right:-20px; bottom:-55px; font-size:12rem; font-weight:700; letter-spacing:-.1em; color:#fff; opacity:.035; }
.eyebrow { font:500 .76rem 'DM Mono',monospace; color:#9db6ff; letter-spacing:.16em; }
.hero h1 { font-size:clamp(2.6rem,5vw,5.3rem); letter-spacing:-.07em; line-height:.95; max-width:820px; margin:.8rem 0 1.2rem; }
.hero h1 em { color:#6fe7dd; font-style:normal; }
.hero p { color:#c3d0e9; font-size:1.18rem; max-width:760px; line-height:1.5; }
.cliff { display:flex; gap:.8rem; align-items:center; margin-top:2rem; font:500 1rem 'DM Mono',monospace; }
.cliff span,.cliff strong { padding:.7rem 1rem; border:1px solid #5272b9; background:#142548; }.cliff strong { color:#6fe7dd; border-style:dashed; }
.section-label { font:600 .75rem 'DM Mono',monospace; color:var(--blue); letter-spacing:.14em; }
.section-title { font-size:2.1rem; letter-spacing:-.05em; margin:.4rem 0 1.2rem; }
.card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1.2rem 1.3rem; height:100%; box-shadow:0 8px 30px rgba(19,35,68,.05); }
.card h3 { margin:.2rem 0 .5rem; }.card p { color:var(--muted); line-height:1.45; }
.step { display:flex; gap:1rem; align-items:flex-start; padding:1rem 0; border-bottom:1px solid #e5e9f1; }.step:last-child { border-bottom:0; }.step-num { min-width:2rem; height:2rem; border-radius:50%; display:grid; place-items:center; background:var(--blue); color:#fff; font-weight:700; }
.code { background:#0b1220; color:#dbe7ff; padding:1.1rem 1.2rem; border-radius:12px; font:500 .84rem/1.5 'DM Mono',monospace; white-space:pre-wrap; }
.status { padding:.8rem 1rem; border-radius:12px; background:#e9fbf7; border:1px solid #bceee5; color:#116e65; font-weight:600; }
.warning { padding:.8rem 1rem; border-radius:12px; background:#fff5d8; border:1px solid #f0d68a; color:#825f04; }
.trace { background:#0b1220; padding:1rem 1.1rem; border-radius:12px; color:#cfe0ff; font:500 .78rem/1.75 'DM Mono',monospace; }
.trace b { color:#6fe7dd; }.footer-note { color:var(--muted); font-size:.8rem; }
div[data-testid='stMetric'] { background:#fff; border:1px solid var(--line); padding:1rem; border-radius:14px; }
</style>
""", unsafe_allow_html=True)

NUMERIC = ["age", "fare", "siblings_spouses", "parents_children"]
CATEGORICAL = ["sex", "passenger_class", "embarked"]


@st.cache_resource
def build_model():
    rng = np.random.default_rng(42)
    n = 900
    data = pd.DataFrame({
        "age": np.clip(rng.normal(30, 14, n), 1, 80),
        "fare": np.round(rng.lognormal(3.1, .75, n), 2),
        "siblings_spouses": rng.integers(0, 5, n),
        "parents_children": rng.integers(0, 4, n),
        "sex": rng.choice(["female", "male"], n),
        "passenger_class": rng.choice(["First", "Second", "Third"], n, p=[.2, .25, .55]),
        "embarked": rng.choice(["C", "Q", "S"], n, p=[.2, .1, .7]),
    })
    age = data.age.fillna(30)
    logit = -.8 + 1.4 * (data.sex == "female") + .9 * (data.passenger_class == "First") + .35 * (data.passenger_class == "Second") - .018 * age + rng.normal(0, .7, n)
    data["survived"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    for column, rate in {"age": .12, "fare": .03, "embarked": .04}.items():
        data.loc[rng.random(n) < rate, column] = np.nan
    X, y = data.drop(columns="survived"), data["survived"]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    pipe = Pipeline([("prepare", prep), ("model", LogisticRegression(max_iter=1000, random_state=42))])
    pipe.fit(X, y)
    return pipe, data


model, training_data = build_model()
if "request_events" not in st.session_state:
    st.session_state.request_events = []

with st.sidebar:
    st.markdown("## ◈ LIVE MLOPS LAB")
    st.caption("One page. One model. One production story.")
    st.divider()
    mode = st.radio("Choose a moment in the lifecycle", ["01 · Notebook cliff", "02 · Training pipeline", "03 · Live inference", "04 · Observe + retrain", "05 · /ml-ops dashboard"], index=2)
    st.divider()
    st.caption("Teaching controls")
    model_version = st.selectbox("Artifact version", ["v1.3 · candidate", "v1.2 · production", "v1.1 · archived"], index=0)
    show_trace = st.toggle("Show request trace", value=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">INDUSTRY SESSION · 2.5 HOURS · LIVE MLOPS</div>
  <h1>We built a model in a notebook.<br><em>What next?</em></h1>
  <p>Turn a happy notebook cell into a versioned, tested, observable prediction service. Students can click through the same moments an ML engineer handles after training.</p>
  <div class="cliff"><span>notebook.ipynb</span><b>→</b><strong>production system</strong></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">THE PRODUCTION JOURNEY</div><div class="section-title">The model is the middle of the story</div>', unsafe_allow_html=True)
journey = st.columns(5)
for col, label, copy in zip(journey, ["DEFINE", "TRAIN", "PACKAGE", "SERVE", "OPERATE"], ["Decision + input contract", "Leak-free pipeline", "Version + release gate", "API + validation", "Signals + retraining"]):
    with col:
        st.markdown(f'<div class="card"><div class="section-label">{label}</div><p>{copy}</p></div>', unsafe_allow_html=True)

if mode == "01 · Notebook cliff":
    st.markdown('<div class="section-label">01 · START WITH THE QUESTION</div><div class="section-title">“It works on my data” is not yet a product</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    for col, title, copy in zip([a, b, c], ["Can another system call it?", "Will tomorrow’s data match?", "How do we know it is healthy?"], ["No schema, endpoint or response contract exists yet.", "Missing values, new categories and drift can change the inputs.", "Accuracy arrives late; errors, latency and drift appear first."]):
        with col:
            st.markdown(f'<div class="card"><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="warning">Teaching move: pause here and ask students to list everything a notebook does not provide to a production caller.</div>', unsafe_allow_html=True)

elif mode == "02 · Training pipeline":
    st.markdown('<div class="section-label">02 · TRAIN SAFELY</div><div class="section-title">Preprocessing + modeling become one deployable object</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown('<div class="card"><h3>Leak-free order</h3><div class="step"><div class="step-num">1</div><div><b>Split</b><br><span class="footer-note">Protect future-like test data</span></div></div><div class="step"><div class="step-num">2</div><div><b>Fit preparation</b><br><span class="footer-note">Learn medians, scaling and categories on train only</span></div></div><div class="step"><div class="step-num">3</div><div><b>Fit model</b><br><span class="footer-note">Estimator sees transformed training features</span></div></div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="code">pipeline = Pipeline([\n  ("prepare", preprocessor),\n  ("model", LogisticRegression())\n])\n\npipeline.fit(X_train, y_train)\npipeline.predict(new_raw_rows)</div>', unsafe_allow_html=True)
    st.markdown('<div class="status">Core idea: the artifact you deploy is preprocessing + model, not the estimator alone.</div>', unsafe_allow_html=True)

elif mode == "03 · Live inference":
    st.markdown('<div class="section-label">03 · SERVE A REAL REQUEST</div><div class="section-title">Send raw JSON. Watch the production path.</div>', unsafe_allow_html=True)
    left, right = st.columns([.9, 1.1])
    with left:
        st.markdown("#### Request body")
        with st.form("prediction_form"):
            age = st.number_input("age", 1, 90, 29)
            fare = st.number_input("fare", 0.0, 1000.0, 85.0)
            sex = st.selectbox("sex", ["female", "male"])
            passenger_class = st.selectbox("passenger_class", ["First", "Second", "Third"])
            embarked = st.selectbox("embarked", ["C", "Q", "S"])
            submitted = st.form_submit_button("POST /predict", type="primary", use_container_width=True)
    with right:
        if submitted:
            payload = {"age": age, "fare": fare, "siblings_spouses": 0, "parents_children": 0, "sex": sex, "passenger_class": passenger_class, "embarked": embarked}
            started = time.perf_counter()
            probability = float(model.predict_proba(pd.DataFrame([payload]))[0, 1])
            latency = (time.perf_counter() - started) * 1000
            st.session_state.request_events.append({
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "status": 200,
                "latency_ms": round(latency, 1),
                "model": model_version.split(" · ")[0],
            })
            st.markdown(f'<div class="status">200 OK · model {model_version.split(" · ")[0]} · {latency:.1f} ms</div>', unsafe_allow_html=True)
            st.metric("Predicted probability", f"{probability:.0%}", "decision = 1" if probability >= .5 else "decision = 0")
            if show_trace:
                trace = ["schema accepted", "missing values checked", "categories encoded", "numbers scaled", "probability calculated", "response logged"]
                st.markdown("<div class='trace'>" + "<br>".join([f"<b>✓</b> {item}" for item in trace]) + "</div>", unsafe_allow_html=True)
        else:
            st.info("Submit the request to see validation, preprocessing, inference, latency and logging.")
    st.markdown('<div class="footer-note">Teaching simulation: the goal is to expose the service boundaries and observability signals, not to make a real survival claim.</div>', unsafe_allow_html=True)

elif mode == "05 · /ml-ops dashboard":
    st.markdown('<div class="section-label">05 · /ML-OPS</div><div class="section-title">Live MLOps command center</div>', unsafe_allow_html=True)
    events = pd.DataFrame(st.session_state.request_events)
    if events.empty:
        events = pd.DataFrame(columns=["time", "status", "latency_ms", "model"])
    errors = int((events.get("status", pd.Series(dtype=int)) >= 400).sum())
    usage_count = len(events)
    p95 = float(events["latency_ms"].quantile(.95)) if not events.empty else 0.0
    avg = float(events["latency_ms"].mean()) if not events.empty else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usage", usage_count, "requests this session")
    c2.metric("Errors", errors, "HTTP 4xx / 5xx")
    c3.metric("p95 latency", f"{p95:.1f} ms" if p95 else "—", "target < 100 ms")
    c4.metric("Drift status", "Watch", "age feature")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Latency and request volume")
        if not events.empty:
            chart = events.reset_index(drop=True)[["latency_ms"]]
            st.line_chart(chart, height=220)
        else:
            st.info("Run predictions from Live inference to populate this dashboard.")
    with right:
        st.markdown("#### Alert rules")
        st.markdown('<div class="card"><p><b>Errors</b> alert if rate &gt; 2%</p><p><b>Latency</b> alert if p95 &gt; 100 ms</p><p><b>Drift</b> alert if PSI &gt; 0.20</p><p><b>Action</b> investigate → rollback → retrain</p></div>', unsafe_allow_html=True)
    st.markdown("#### Recent structured logs")
    if not events.empty:
        st.dataframe(events.sort_index(ascending=False).head(10), use_container_width=True, hide_index=True)
    else:
        st.caption("No events yet. Logs will appear here after the first prediction.")
    st.markdown('<div class="warning">Session-local teaching dashboard: production systems should send these events to a durable log store and metrics backend.</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="section-label">04 · OPERATE THE SYSTEM</div><div class="section-title">Production quality can change while the code stays the same</div>', unsafe_allow_html=True)
    events = pd.DataFrame(st.session_state.request_events)
    usage_count = len(events)
    avg_latency = f"{events['latency_ms'].mean():.1f} ms" if not events.empty else "—"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usage", usage_count, "requests this session")
    c2.metric("p95 latency", avg_latency, "live trace" if not events.empty else "no requests yet")
    c3.metric("Errors", "0", "schema + 5xx")
    c4.metric("Age drift", "Alert", "investigate", delta_color="inverse")
    if not events.empty:
        st.markdown("#### Recent request log")
        st.dataframe(events.sort_index(ascending=False).head(8), use_container_width=True, hide_index=True)
    st.markdown("#### Drift does not mean retrain immediately")
    st.dataframe(pd.DataFrame({"signal": ["input drift", "invalid requests", "label quality", "latency"], "what it tells us": ["population changed", "caller contract failing", "model quality changed", "service degraded"], "next action": ["investigate", "fix/communicate schema", "compare candidate", "inspect service"],}), use_container_width=True, hide_index=True)
    st.markdown('<div class="warning">Retraining loop: detect → investigate → retrain → compare → canary → promote or rollback.</div>', unsafe_allow_html=True)

st.divider()
st.caption("Notebook → production is a system design conversation. The pipeline is the bridge; tests, contracts and monitoring keep the bridge standing.")
