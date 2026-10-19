# Spam Detection ML Inference API

A REST API for classifying text messages as **spam** or **not spam** using a trained PyTorch neural network and TF-IDF text vectorization. The API is built with FastAPI and packaged with Docker.

## Features

- Text classification using a trained deep learning model
- TF-IDF text vectorization
- REST API built with FastAPI
- Input validation using Pydantic
- Health-check endpoint
- Interactive Swagger API documentation
- Automated tests using pytest
- Docker containerization

## Tech Stack

- Python 3.12
- PyTorch
- Scikit-learn
- TF-IDF
- FastAPI
- Uvicorn
- Pytest
- Docker

## Project Structure

```text
Task_06_ML_Inference_API/
├── main.py
├── model.py
├── spam_classifier.pth
├── tfidf_vectorizer.pkl
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── tests/
    └── test_main.py
```

## How It Works

1. The user submits a text message to the `/predict` endpoint.
2. The TF-IDF vectorizer converts the text into numerical features.
3. The trained PyTorch neural network calculates the spam probability.
4. The API returns the predicted class and probability as JSON.

## Run Locally

### 1. Install dependencies

Create and activate a Python virtual environment if you do not already have one, then run:

```bash
python -m pip install -r requirements.txt
```

### 2. Start the API

```bash
python -m uvicorn main:app --reload
```

### 3. Open the API documentation

Visit:

http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict whether text is spam |

### Prediction Request

```json
{
  "text": "Congratulations! You have won a free prize."
}
```

### Example Response

```json
{
  "text": "Congratulations! You have won a free prize.",
  "prediction": "spam",
  "probability": 0.9923
}
```

The probability above is an illustrative example. Actual output depends on the model and input text.

## Run Automated Tests

Run the following command from the project root:

```bash
python -m pytest tests/test_main.py -v
```

The tests cover API endpoints, prediction responses, input validation, and invalid routes.

## Run with Docker

### 1. Build the image

```bash
docker build -t spam-detection-api .
```

### 2. Start the container

```bash
docker run --rm -p 8000:8000 spam-detection-api
```

### 3. Test the API

Open:

http://127.0.0.1:8000/docs

To stop the container, press `Ctrl + C` in the terminal running it.

## Model Artifacts

- `spam_classifier.pth`: trained PyTorch model weights
- `tfidf_vectorizer.pkl`: fitted TF-IDF vectorizer

The model architecture in `model.py` must match the architecture used when the weights were saved.

## Limitations

Predictions depend on the training data and model quality. The API is intended as a demonstration project and should be evaluated on representative data before production use.

## Author

Developed as part of a Machine Learning Inference REST API internship/capstone task.
