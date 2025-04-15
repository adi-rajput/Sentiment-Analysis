# sentiment_benchmark.py

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm import tqdm
import torch

# ------------------------------------------------------------
# Step 1: Configuration
# ------------------------------------------------------------

# Transformer-based model configs
transformer_models = {
    "DistilBERT (SST-2)": "distilbert-base-uncased-finetuned-sst-2-english",
    "BERT (Multilingual - 5 Stars)": "nlptown/bert-base-multilingual-uncased-sentiment",
    "RoBERTa (Twitter)": "cardiffnlp/twitter-roberta-base-sentiment"
}

# Traditional ML models
traditional_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVC": LinearSVC()
}

# ------------------------------------------------------------
# Step 2: Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("data.csv")  # Ensure 'text' and 'label' columns exist
texts = df['text'].tolist()
true_labels = df['label'].tolist()

# ------------------------------------------------------------
# Step 3: Helper Functions
# ------------------------------------------------------------

def normalize_prediction(model_name, result):
    """
    Normalize prediction label to binary:
    1 -> Positive, 0 -> Negative
    """
    label = result['label'].lower()
    if "sst-2" in model_name or "twitter" in model_name:
        return 1 if "pos" in label else 0
    elif "nlptown" in model_name:
        stars = int(label[0])
        return 1 if stars >= 4 else 0
    return 0

def evaluate_model(name, y_true, y_pred):
    """
    Compute all metrics for a given model.
    """
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Precision": round(precision_score(y_true, y_pred), 3),
        "Recall": round(recall_score(y_true, y_pred), 3),
        "F1 Score": round(f1_score(y_true, y_pred), 3)
    }

def print_prediction_output(text, pred, score):
    """
    Print individual prediction (optional, useful for debugging).
    """
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Text       : {text}")
    print(f"Prediction : {sentiment} (Confidence: {score})\n")

# ------------------------------------------------------------
# Step 4: Run Transformer Benchmarks
# ------------------------------------------------------------

benchmark_results = []
model_predictions = {}

print("\n🔬 Running Transformer-based Sentiment Models\n")

for model_desc, model_name in transformer_models.items():
    print(f"🔍 Model: {model_desc}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False,
        device=0 if torch.cuda.is_available() else -1
    )

    preds = []
    for text in tqdm(texts, desc=f"Inference with {model_desc}"):
        result = pipe(text)[0]
        pred = normalize_prediction(model_name, result)
        preds.append(pred)
        # Uncomment for sentence-wise results
        # print_prediction_output(text, pred, round(result['score'], 2))

    model_predictions[model_desc] = preds
    metrics = evaluate_model(model_desc, true_labels, preds)
    benchmark_results.append(metrics)

# ------------------------------------------------------------
# Step 5: Run Traditional ML Benchmarks
# ------------------------------------------------------------

print("\n🔬 Running Traditional ML Models\n")

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
y = true_labels

for model_desc, clf in traditional_models.items():
    print(f"🔍 Model: {model_desc}")
    print("-" * 60)

    clf.fit(X, y)
    preds = clf.predict(X)

    model_predictions[model_desc] = preds
    metrics = evaluate_model(model_desc, y, preds)
    benchmark_results.append(metrics)

# ------------------------------------------------------------
# Step 6: Display Benchmark Summary
# ------------------------------------------------------------

print("\n📊 Final Model Benchmark Summary\n")

df_results = pd.DataFrame(benchmark_results)
print(df_results.to_string(index=False))

# ------------------------------------------------------------
# Step 7: Recommend Best Model
# ------------------------------------------------------------

best_model = df_results.sort_values(by="F1 Score", ascending=False).iloc[0]
print(f"\n🏆 Best Performing Model: {best_model['Model']} with F1 Score = {best_model['F1 Score']}")
