import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from preprocess import clean_text, lemmatize_text
from feature_engineering import build_tfidf, linguistic_features
import numpy as np

def train():

    df = pd.read_csv("data/mypersonality_sample.csv")
    
    df["text"] = df["text"].apply(clean_text)
    df["text"] = df["text"].apply(lemmatize_text)

    X_tfidf, vectorizer = build_tfidf(df["text"])
    X_ling = linguistic_features(df["text"])

    X = np.hstack((X_tfidf.toarray(), X_ling))

    traits = ["OPN","CON","EXT","AGR","NEU"]

    os.makedirs("models", exist_ok=True)

    for trait in traits:
        y = df[trait]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        models = {
            "lr": LogisticRegression(max_iter=1000),
            "rf": RandomForestClassifier(),
            "xgb": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        best_model = None
        best_score = 0

        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            if score > best_score:
                best_score = score
                best_model = model

        joblib.dump(best_model, f"models/{trait}_model.pkl")

    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Training completed for all traits.")