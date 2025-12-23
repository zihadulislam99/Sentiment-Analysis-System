from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "tabularisai/multilingual-sentiment-analysis"
save_dir = "sentiment_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
