from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "ticket_classifier.joblib"

TRAINING_DATA= [
    #Technical problems
    ("The app crashes whenever I open it", "technical"),
    ("the website shows a server error","technical"),
    ("I cannot upload my document","technical"),
    ("The page keeps freezing on my computer", "technical"),
    ("The application is running very slowly","technical"),
    ("I receive an error when I submit the form","technical"),
    ("The mobile app closes unexpectedly", "technical"),
    ("The website is not loading correctly","technical"),
    ("I cannot download my invoice PDF", "technical"),
    ("The search function does not work","technical"),
    ("The system keeps crashing during checkout", "technical"),

    # Billing Problems
    ("I was charged twice for the same order","billing"),
    ("Please explain this unexpected charge","billing"),
    ("I need a refund for my payment","billing"),
    ("My invoice contains the wrong amount","billing"),
    ("Why was my subscription renewed", "billing"),
    ("My payment was declined but money was deducted", "billing"),
    ("I want to update my billing information", "billing"),
    ("There is an extra fee on my account", "billing"),
    ("Please send me a copy of my receipt", "billing"),
    ("I do not recognize this transaction", "billing"),
    ("My refund has not arrived yet", "billing"),
    ("I need help understanding my monthly bill", "billing"),

  # Account access
    ("I forgot my password and cannot sign in", "account_access"),
    ("The password reset email never arrived", "account_access"),
    ("My account has been locked", "account_access"),
    ("I cannot verify my email address", "account_access"),
    ("I need help changing my password", "account_access"),
    ("Two factor authentication is not working", "account_access"),
    ("I am unable to log into my account", "account_access"),
    ("My verification code keeps getting rejected", "account_access"),
    ("I lost access to my registered email", "account_access"),
    ("Please help me unlock my profile", "account_access"),
    ("My login credentials are not accepted", "account_access"),
    ("I cannot access my account after resetting my password", "account_access"),

    # Order status
    ("Where is my order right now", "order_status"),
    ("My package has not arrived", "order_status"),
    ("Can you provide my tracking number", "order_status"),
    ("My shipment is delayed", "order_status"),
    ("The tracking page has not updated", "order_status"),
    ("When will my order be delivered", "order_status"),
    ("My order says shipped but is not here", "order_status"),
    ("I want an update on my delivery", "order_status"),
    ("The delivery date has passed", "order_status"),
    ("Please check the status of my purchase", "order_status"),
    ("My package is still in transit", "order_status"),
    ("I received a shipping confirmation but no package", "order_status"),
]

def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1,2),
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

def main() -> None:
    texts = [text for text, label in TRAINING_DATA]
    labels = [label for text, label in TRAINING_DATA]

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,

    )

    evaluation_model = build_model()
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)

    print("\nEvaluation on held-out example tickets:")
    print(classification_report(y_test, predictions, zero_division=0))

    #Train the final model on all available examples.
    final_model = build_model()
    final_model.fit(texts, labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    print(f"\nTraining examples: {len(texts)}") #used to return the number of items in an object, such as characters in a string, elements in a list or tuple, or keys in a dictionary.
    print(f"categories: {sorted(set(labels))}")
    print(f"Saved model to:{MODEL_PATH}")

if __name__=="__main__":
    main()