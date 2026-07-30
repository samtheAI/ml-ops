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

1. Create a GitHub repository and push this folder.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Choose **Create app**, then select the repository, branch and `app.py`.
4. Choose Python 3.12 in Advanced settings if prompted.
5. Deploy.

Streamlit Community Cloud reads `requirements.txt` from the repository and installs the pinned dependencies.

## Teaching flow

- Notebook cliff: what a notebook does not provide.
- Training pipeline: preprocessing, modeling and leakage prevention.
- Live inference: raw request → validation → transformation → prediction → trace.
- Observe + retrain: operational signals, drift, investigation and rollback.
