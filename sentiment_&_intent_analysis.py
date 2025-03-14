# -*- coding: utf-8 -*-
"""Sentiment & Intent Analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1d1L1DidR0MR-p4HIUAPFLQhory4w1-s1
"""

!pip install transformers torch

from transformers import pipeline

# Load a sentiment analysis pipeline
sentiment_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Transcribed conversation (Patient responses only)
patient_responses = [
    "Good morning, doctor. I’m doing better, but I still have some discomfort now and then.",
    "Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.",
    "Yes, I always do.",
    "At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.",
    "Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home.",
    "The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.",
    "It’s not constant, but I do get occasional backaches. It’s nothing like before, though.",
    "No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.",
    "I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.",
    "That’s a relief!",
    "That’s great to hear. So, I don’t need to worry about this affecting me in the future?",
    "Thank you, doctor. I appreciate it."
]

# Function to classify sentiment
def classify_sentiments(responses):
    results = {}
    for idx, response in enumerate(responses):
        sentiment_result = sentiment_classifier(response)[0]["label"]
        sentiment_mapping = {
            "NEGATIVE": "Anxious",
            "POSITIVE": "Reassured"
        }
        sentiment = sentiment_mapping.get(sentiment_result, "Neutral")
        results[f"Response {idx+1}"] = {"Text": response, "Sentiment": sentiment}
    return results

# Apply sentiment classification
sentiment_analysis_results = classify_sentiments(patient_responses)


import json

# Print formatted output with correct encoding
print(json.dumps(sentiment_analysis_results, indent=4, ensure_ascii=False))

