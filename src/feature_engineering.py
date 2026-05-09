import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(corpus):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1,2)
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def linguistic_features(texts):
    features = []
    for text in texts:
        words = text.split()
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        word_count = len(words)
        features.append([avg_word_length, word_count])
    return np.array(features)