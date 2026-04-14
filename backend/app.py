from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os

app = Flask(__name__)
CORS(app)

# ==============================
# Load Model (DistilBERT)
# ==============================

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(base_dir, "..", "models", "bert_model"))

tokenizer = DistilBertTokenizer.from_pretrained(model_path, local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# ==============================
# Rule-Based Fix (IMPORTANT)
# ==============================

def rule_based_fix(text):
    text_lower = text.lower()

    if (
        "confirmed to be false" in text_lower
        or "is false" in text_lower
        or "not true" in text_lower
        or "rumor is false" in text_lower
        or "fake news" in text_lower
    ):
        return "REAL", 0.95

    return None, None

# ==============================
# BERT Prediction
# ==============================

def predict_bert(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()

    return pred, confidence

# ==============================
# Routes
# ==============================

@app.route("/")
def home():
    return "AI Fake News Detector (BERT) Running 🚀"

@app.route("/predict_text", methods=["POST"])
def predict_text():
    data = request.json
    text = data.get("text", "")

    # 🔥 Rule-based override
    rule_result, rule_conf = rule_based_fix(text)

    if rule_result:
        return jsonify({
            "prediction": rule_result,
            "confidence": round(rule_conf * 100, 2),
            "source": "Rule-based"
        })

    # 🤖 BERT prediction
    pred, confidence = predict_bert(text)
    result = "FAKE" if pred == 1 else "REAL"

    return jsonify({
        "prediction": result,
        "confidence": round(confidence * 100, 2),
        "source": "BERT"
    })

# ==============================
# Run App
# ==============================

if __name__ == "__main__":
    app.run(debug=True)