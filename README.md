# ARG_Dashboard (Self-Contained)

This folder is self-contained and can be moved to a different directory or machine.

## Structure

- `code/`
  - `app.py`
  - `data_utils.py`
  - `pages/1_Opportunity_Analysis.py`
  - `requirements.txt`
  - `data_processing.ipynb`
- `data/`
  - `input/`
  - `intermediate/`

## Run Locally

From `ARG_Dashboard/code`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

- Set app path to: `code/app.py`
- Keep folder layout as-is so the app can resolve:
  - `../data/input`
  - `../data/intermediate`

## Notes

- The app is configured for Argentina (`FOCUS_ISO = "ARG"`).
- Main cached metrics file used by default:
  - `data/intermediate/opportunity_metrics_hs4_arg.csv`
- For GitHub deployability, very large raw files are excluded from version control:
  - `data/input/hs92_country_country_product_year_6_2020_2024.csv`
  - `data/intermediate/complexity_calculations.csv`
- The app uses the lightweight file `data/intermediate/complexity_arg_2024.csv` for production runtime.
