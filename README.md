# Notebook → Production · Live MLOps Lab

One-page Streamlit teaching app for a 2.5-hour session on production-ready ML pipelines.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Choose **Create app**.
3. Select repository `samtheAI/ml-ops`, branch `main`, and file `app.py`.
4. Choose Python 3.12 in Advanced settings if prompted.
5. Deploy.

Community Cloud reads `requirements.txt` and installs the pinned dependencies.

## Deployment checklist

- Confirm `app.py` runs locally with `streamlit run app.py`.
- Keep the final pipeline and its preprocessing together in the app artifact.
- Validate the raw request before calling `predict_proba`.
- Check the Streamlit Cloud logs after the first deploy.
- Use the app's **Observe + retrain** mode to inspect usage, latency, errors and drift.
- Treat a drift alert as an investigation trigger; compare a candidate model before promoting it.

## Production extension points

This teaching app uses session-local metrics so students can see the live flow. A real service would send structured request events to a durable log or metrics backend, then add alerting, retention, access controls and a controlled retraining pipeline.

## Teaching flow

- Notebook cliff: what a notebook does not provide.
- Training pipeline: preprocessing, modeling and leakage prevention.
- Live inference: raw request → validation → transformation → prediction → trace.
- Observe + retrain: operational signals, drift, investigation and rollback.
