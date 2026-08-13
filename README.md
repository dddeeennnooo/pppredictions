# Serie A BTTS Predictor

A Python project for predicting football match outcomes, with a baseline logistic
regression model for whether both teams will score (BTTS).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Commands

Initialize and populate the database, then build the pre-match features:

```powershell
python -m src.main init-db
python -m src.main import-serie-a
python -m src.main build-features
```

Train and evaluate the BTTS model using a chronological 80/20 split:

```powershell
python -m src.main train-btts
```

The local SQLite database, virtual environment, and generated model artifacts are
excluded from version control.
