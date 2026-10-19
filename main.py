
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model import predict_spam

app = FastAPI(
    title="Spam Detection ML API",
    description="A REST API powered by a trained PyTorch NLP classifier.",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The text message to classify"
    )


class PredictionResponse(BaseModel):
    text: str
    prediction: str
    probability: float = Field(ge=0.0, le=1.0)


@app.get("/")
def home():
    return {
        "message": "Spam Detection ML API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        result = predict_spam(request.text)

        return PredictionResponse(
            text=request.text,
            prediction=result["prediction"],
            probability=result["probability"]
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
