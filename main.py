import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch

# Model configurations
model_configs = {
    "DistilBERT (SST-2)": "distilbert-base-uncased-finetuned-sst-2-english",
    "BERT (Multilingual - 5 Stars)": "nlptown/bert-base-multilingual-uncased-sentiment",
    "RoBERTa (Twitter)": "cardiffnlp/twitter-roberta-base-sentiment"
}

# Load dataset
df = pd.read_csv("data.csv")
texts = df['text'].tolist()
true_labels = df['label'].tolist()  # 0 = Negative, 1 = Positive

# Normalize predictions for different label formats
def normalize_prediction(model_name, result):
    label = result['label'].lower()
    if "sst-2" in model_name or "twitter" in model_name:
        return 1 if "pos" in label else 0
    elif "nlptown" in model_name:
        stars = int(label[0])
        return 1 if stars >= 4 else 0
    return 0

# Store benchmark metrics
benchmark_results = []

# For storing model-wise predictions
model_predictions = {}

print("\n📊 Sentiment Predictions Per Model:\n")

for model_desc, model_name in model_configs.items():
    print(f"🔍 Model: {model_desc}\n{'-' * 50}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipeline_model = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False,
        device=0 if torch.cuda.is_available() else -1
    )

    preds = []
    per_text_output = []

    for text in texts:
        result = pipeline_model(text)[0]
        pred = normalize_prediction(model_name, result)
        label = "Positive" if pred == 1 else "Negative"
        score = round(result['score'], 2)
        preds.append(pred)
        per_text_output.append((text, label, score))

    # Print individual sentence predictions
    for text, label, score in per_text_output:
        print(f"Text     : {text}")
        print(f"Prediction: {label} (Confidence: {score})\n")

    # Save predictions for benchmark
    model_predictions[model_desc] = preds

    # Evaluate metrics
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds)
    rec = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)

    benchmark_results.append({
        "Model": model_desc,
        "Accuracy": round(acc, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1 Score": round(f1, 2)
    })

# Create formal benchmark summary
print("\n📈 Model Benchmark Summary:\n")
df_benchmark = pd.DataFrame(benchmark_results)
print(df_benchmark.to_string(index=False))
