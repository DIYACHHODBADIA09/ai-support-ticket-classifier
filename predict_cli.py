
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ticket_classifier.joblib"


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run: python train_model.py")
        return

    model = joblib.load(MODEL_PATH)

    print("\nAI Support Ticket Classifier")
    print("Type 'quit' to exit.\n")

    while True:
        text = input("Enter a support ticket: ").strip()

        if text.lower() == "quit":
            print("Goodbye!")
            break

        if len(text) < 5:
            print("Please enter at least 5 characters.")
            continue

        prediction = model.predict([text])[0]
        print(f"Predicted category: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            confidence = max(probabilities)
            print(f"Model confidence: {confidence:.1%}")

        print()


if __name__ == "__main__":
    main()
