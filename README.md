# Text-Based Personality Prediction System

A Machine Learning and Natural Language Processing (NLP) based system that predicts personality traits from textual input using the Big Five Personality Model.

---

# 📌 Overview

This project analyzes textual data and predicts personality traits automatically using NLP and Machine Learning techniques. The system processes user input text, extracts meaningful linguistic features, and classifies personality traits based on the Big Five Personality Model:

* Openness (OPN)
* Conscientiousness (CON)
* Extraversion (EXT)
* Agreeableness (AGR)
* Neuroticism (NEU)

The project demonstrates the practical application of text preprocessing, feature extraction, supervised learning, and personality analysis using Python.

---

# 🚀 Features

* Text preprocessing using NLP techniques
* TF-IDF feature extraction
* Personality prediction using Machine Learning
* Multiple classification models:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Separate prediction models for each personality trait
* Model training and evaluation pipeline
* Prediction on unseen user text input

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Libraries & Tools

* Pandas
* NumPy
* Scikit-learn
* NLTK
* spaCy
* Joblib
* XGBoost

## NLP Techniques

* Tokenization
* Stopword Removal
* Lemmatization
* TF-IDF Vectorization

---

# 📂 Project Structure

```text
TextPersonalityPrediction/
│
├── data/
│   ├── data-final.csv
│   └── mypersonality_sample.csv
│
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── predict.py
│
├── models/
│
├── results/
│
├── convert_dataset.py
├── main.py
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/TextPersonalityPrediction.git
cd TextPersonalityPrediction
```

---

## Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk spacy xgboost joblib
```

---

## Download NLP Resources

```bash
python -m spacy download en_core_web_sm
```

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

---

# 📊 Dataset

The project uses the Big Five Personality Dataset (`data-final.csv`).

The dataset contains personality-related responses which are converted into text-based representations for NLP processing.

For faster training and testing, a subset of the dataset can be sampled during preprocessing.

---

# 🔄 Workflow

1. Dataset preprocessing
2. Text cleaning and normalization
3. TF-IDF feature extraction
4. Train-test split
5. Model training
6. Model evaluation
7. Personality prediction

---

# 🤖 Machine Learning Models

The following algorithms were used:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

Each personality trait is treated as a separate binary classification problem.

---

# ▶️ How to Run

## Step 1 — Convert Dataset

```bash
python convert_dataset.py
```

---

## Step 2 — Train Models

```bash
python main.py
```

Choose:

```text
1. Train Models
```

---

## Step 3 — Predict Personality

Run again:

```bash
python main.py
```

Choose:

```text
2. Predict Personality
```

Enter custom text input for prediction.

---

# 📈 Evaluation Metrics

The system uses:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix

for evaluating model performance.

---

# 💡 Sample Input

```text
I enjoy meeting new people and exploring new ideas.
```

## Sample Output

```text
OPN: High
CON: Medium
EXT: High
AGR: High
NEU: Low
```

---

# 🔮 Future Improvements

* Transformer-based personality prediction using BERT
* Deep learning implementation (LSTM)
* Real-time GUI/Desktop application
* Social media integration
* Multilingual personality prediction

---

# 📚 Learning Outcomes

This project helped in understanding:

* Natural Language Processing
* Text Vectorization
* Machine Learning Pipelines
* Personality Prediction Systems
* Model Evaluation Techniques
* Real-world NLP workflows

---

# 👨‍💻 Author

**Kailash NP**

Project: Text-Based Personality Prediction using NLP and Machine Learning.

