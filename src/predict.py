import joblib
import numpy as np
from preprocess import clean_text, lemmatize_text
from feature_engineering import linguistic_features

def predict_personality(text):

    vectorizer = joblib.load("models/vectorizer.pkl")
    traits = ["OPN","CON","EXT","AGR","NEU"]

    text = clean_text(text)
    text = lemmatize_text(text)

    X_tfidf = vectorizer.transform([text])
    X_ling = linguistic_features([text])
    X = np.hstack((X_tfidf.toarray(), X_ling))

    results = {}

    for trait in traits:
        model = joblib.load(f"models/{trait}_model.pkl")
        prediction = model.predict(X)[0]
        results[trait] = "High" if prediction == 1 else "Low"

    return results
