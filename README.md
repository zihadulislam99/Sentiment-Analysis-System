[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-purple.svg)](#)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](CONTRIBUTING.md)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

# Multilingual Sentiment Analysis System

A **multilingual text sentiment classification system** built with **Python, PyTorch, and Hugging Face Transformers**.
The system predicts sentiment across **five levels** — *Very Negative, Negative, Neutral, Positive, Very Positive* — and supports **20+ languages**, including **English and Bengali (বাংলা)**.
It runs fully **offline** using locally stored model files, making it suitable for secure, low-connectivity, or production environments.

---

## **Features**

* **Multilingual Sentiment Analysis:** Supports over 20 global languages.
* **5-Class Classification:** Very Negative → Very Positive.
* **Offline Inference:** No internet required during prediction.
* **Batch & Single Text Support:** Analyze one or multiple texts at once.
* **Transformer-Based Model:** High accuracy using modern NLP architectures.
* **Lightweight Inference Mode:** Uses `model.eval()` and `torch.no_grad()`.
* **Easy Integration:** Can be plugged into APIs, web apps, or data pipelines.

---

## **Task Details**

| Property              | Description                                               |
| --------------------- | --------------------------------------------------------- |
| **Task**              | Text Classification (Sentiment Analysis)                  |
| **Number of Classes** | 5                                                         |
| **Labels**            | Very Negative, Negative, Neutral, Positive, Very Positive |
| **Framework**         | PyTorch                                                   |
| **Model Type**        | Transformer (Hugging Face)                                |
| **Inference Mode**    | Offline / Local                                           |

---

## **Supported Languages**

English, 中文 (Chinese), Español (Spanish), हिन्दी (Hindi), العربية (Arabic), বাংলা (Bengali), Português (Portuguese), Русский (Russian), 日本語 (Japanese), Deutsch (German), Bahasa Melayu (Malay), తెలుగు (Telugu), Tiếng Việt (Vietnamese), 한국어 (Korean), Français (French), Türkçe (Turkish), Italiano (Italian), Polski (Polish), Українська (Ukrainian), Tagalog, Nederlands (Dutch), Schweizerdeutsch (Swiss German), Kiswahili (Swahili)

---

## **Technology Stack**

* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch
* **NLP Library:** Hugging Face Transformers
* **Tokenizer:** AutoTokenizer
* **Model Loader:** AutoModelForSequenceClassification

---

## **Project Structure**

```
Sentiment-Analysis-System/
│
├── sentiment_model/           # Local trained model & tokenizer files
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab files
│
├── inference.py                   # Sentiment prediction script
└── README.md                      # Project documentation
```

---

## **Setup Instructions**

### 1. Install Dependencies

```bash
pip install torch transformers
```

> ⚠️ Internet is **not required** during inference if the model is already stored locally.

---

### 2. Model Preparation

Ensure your trained model and tokenizer are available locally at:

```
./sentiment_model
```

The system loads the model using:

```python
local_files_only=True
```

---

### 3. Run Sentiment Prediction

Create or run the inference script:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "./sentiment_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

def predict_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    sentiment_map = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive"
    }

    return [sentiment_map[i] for i in torch.argmax(probs, dim=-1).tolist()]
```

---

## **Usage Example**

```python
print(predict_sentiment("সেবার মান একেবারেই খারাপ।")[0])
```

**Output**

```
Very Negative
```

```python
texts = ["Ti amo", "I loVe you"]
print(predict_sentiment(texts))
```

**Output**

```
['Very Positive', 'Very Positive']
```

---

## **Tips for Better Results**

* Keep text length under **512 tokens**.
* Use clear, complete sentences for higher accuracy.
* Batch processing improves performance for large datasets.
* Ideal for news, reviews, and social media content.

---

## **Applications**

* News sentiment analysis
* Social media monitoring
* Customer feedback analysis
* Multilingual NLP pipelines
* AI-based decision and risk analysis systems
* Offline or secure environments

---

## **License**

This project is **open-source** and free to use for **personal, educational, and research purposes**.

---

## **Author**

**Zihadul Islam**
GitHub: [https://github.com/zihadulislam99](https://github.com/zihadulislam99)
