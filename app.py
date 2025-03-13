from fastapi import FastAPI
import spacy
from transformers import pipeline
from pydantic import BaseModel

# Create FastAPI instance
app = FastAPI()

# Load models
nlp = spacy.load("en_core_web_sm")
sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define input model
class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Physician Notetaker API is running!"}

@app.post("/analyze/")
def analyze_text(data: InputText):
    text = data.text

    # Named Entity Recognition (NER)
    doc = nlp(text)
    symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]

    # Sentiment Analysis
    sentiment = sentiment_model(text)[0]['label']

    # Example SOAP Note Structure
    soap_note = {
        "Subjective": {"Chief_Complaint": text},
        "Objective": {"Findings": symptoms},
        "Assessment": {"Sentiment": sentiment},
        "Plan": {"Next Steps": "Continue observation or follow-up if needed"}
    }

    return {"Symptoms": symptoms, "Sentiment": sentiment, "SOAP Note": soap_note}

# This ensures the script runs only when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
