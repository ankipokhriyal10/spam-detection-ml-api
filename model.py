
import os
import joblib
import torch
import torch.nn as nn


class SpamClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "spam_classifier.pth")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

device = torch.device("cpu")

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load(VECTORIZER_PATH)

# Recreate the architecture using the vectorizer's feature count
input_size = len(vectorizer.get_feature_names_out())

model = SpamClassifier(input_size=input_size)

# Load the saved trained weights
state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()


def predict_spam(text: str) -> dict:
    """Predict whether a text message is spam or legitimate."""

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    # Convert text into TF-IDF features
    features = vectorizer.transform([text]).toarray()

    # Convert features into a PyTorch tensor
    input_tensor = torch.tensor(
        features,
        dtype=torch.float32,
        device=device
    )

    # Get probability using sigmoid because the model has one output logit
    with torch.inference_mode():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    # Assumes the original training labels were 1 = spam and 0 = not spam
    prediction = "spam" if probability >= 0.5 else "not_spam"

    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }
