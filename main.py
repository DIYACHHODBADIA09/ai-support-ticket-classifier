
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ticket_classifier.joblib"
DATABASE_PATH = BASE_DIR / "tickets.db"

engine = create_engine(
    f"sqlite:///{DATABASE_PATH.as_posix()}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    pass


class TicketRecord(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_category: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class TicketInput(BaseModel):
    text: str = Field(..., min_length=5, max_length=2000)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("Ticket text must contain at least 5 characters.")
        return value


classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run train_model.py first."
        )

    classifier = joblib.load(MODEL_PATH)
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="AI Support Ticket Classifier",
    description="Classify support tickets using machine learning.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "message": "AI Support Ticket Classifier API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
    }


@app.post("/predict")
def predict_ticket(
    ticket: TicketInput,
    db: Session = Depends(get_db),
):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="The machine learning model is not loaded.",
        )

    try:
        predicted_category = str(classifier.predict([ticket.text])[0])
        probabilities = classifier.predict_proba([ticket.text])[0]
        confidence = float(max(probabilities))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not classify this ticket.",
        ) from exc

    record = TicketRecord(
        text=ticket.text,
        predicted_category=predicted_category,
        confidence=confidence,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "text": record.text,
        "predicted_category": record.predicted_category,
        "confidence": record.confidence,
        "created_at": record.created_at,
    }


@app.get("/tickets")
def get_tickets(db: Session = Depends(get_db)):
    records = (
        db.query(TicketRecord)
        .order_by(TicketRecord.id.desc())
        .all()
    )

    return [
        {
            "id": record.id,
            "text": record.text,
            "predicted_category": record.predicted_category,
            "confidence": record.confidence,
            "created_at": record.created_at,
        }
        for record in records
    ]
