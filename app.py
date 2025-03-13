import spacy
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
import subprocess

# Ensure the spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Create FastAPI app
app = FastAPI()

# Load sentiment analysis model
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

    return {"Symptoms": symptoms, "Sentiment": sentiment}
