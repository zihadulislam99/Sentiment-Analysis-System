from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# If the local "sentiment_model" is not working, then you need to run this "Download.py" file first.
MODEL_PATH = "./Sentiment_Analysis/sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()  # important for inference

def predict_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    # sentiment_map = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
    return [sentiment_map[i] for i in torch.argmax(probs, dim=-1).tolist()]

texts = ["Ti amo", "I loVe you"]

print(predict_sentiment("সেবার মান একেবারেই খারাপ।")[0])
print(predict_sentiment(["Ti amo", "I loVe you"]))
for text, sentiment in zip(texts, predict_sentiment(texts)):

    print(f"Text: {text}\nSentiment: {sentiment}\n")
