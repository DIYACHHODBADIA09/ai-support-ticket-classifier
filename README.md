# AI Support Ticket Classifier


# AI Support Ticket Classifier

A beginner machine-learning project that classifies customer support
tickets into four categories using Python and scikit-learn.

## Features

- Text classification using TF-IDF and Logistic Regression
- FastAPI endpoints for predictions and health checks
- SQLite storage for ticket history
- Input validation with Pydantic
- Automated tests with pytest

## Categories

- `account_access`
- `billing`
- `technical`
- `order_status`

## Requirements

Python 3.11 or newer is recommended.

## Setup on Windows

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train and save the model:

```powershell
python train_model.py
```

Run tests:

```powershell
python -m pytest -v
```

Start the API:

```powershell
python -m uvicorn main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

- `GET /` — API welcome message
- `GET /health` — application and model status
- `POST /predict` — classify and save a support ticket
- `GET /tickets` — retrieve saved ticket history

## Example Request

```json
{
  "text": "I cannot log in to my account"
}
```

## Limitations

This project uses a small demonstration dataset. Predictions and
confidence scores are not production-ready. Evaluate the model on
representative, unseen tickets before using it for real customer support.

## Technologies

Python, scikit-learn, TF-IDF, Logistic Regression, FastAPI,
SQLAlchemy, SQLite, Pydantic, pytest.
