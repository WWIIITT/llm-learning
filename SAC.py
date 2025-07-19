import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample sentiment data"""
    positive_texts = [
    "I love this movie! It's fantastic.",
    "Great acting and amazing story.",
    "Wonderful experience, highly recommend.",
    "Best film I've seen this year.",
    "Absolutely brilliant and entertaining."
    ]

    negative_texts = [
    "Terrible movie, waste of time.",
    "Boring plot and bad acting.",
    "Completely disappointed.",
    "Worst film ever made.",
    "Awful in every way possible."
    ]

    texts = positive_texts*20 + negative_texts*20
    labels = [1]* 100 + [0]*100

    return pd.DataFrame({'text': texts, 'label': labels})

df = create_sample_data()
print(f'dataframe shape: {df.shape}')
print(f'Label distribution:\n{df["label"].value_counts()}')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f'Train shape: {len(train_df)}, Test shape: {len(test_df)}')

from transformers import (
AutoTokenizer,
AutoModelForSequenceClassification,
TrainingArguments,
Trainer,
DataCollatorWithPadding
)
import torch

model_name = "distilbert-base-uncased" # Smaller, faster than BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=128)

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format
train_dataset = train_dataset.remove_columns(['text'])
test_dataset = test_dataset.remove_columns(['text'])
train_dataset.set_format('torch')
test_dataset.set_format('torch')


from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)

    return {
    'accuracy': accuracy,
    'f1': f1,
    'precision': precision,
    'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
output_dir='./results',
num_train_epochs=3,
per_device_train_batch_size=16,
per_device_eval_batch_size=64,
warmup_steps=500,
weight_decay=0.01,
logging_dir='./logs',
logging_steps=10,
eval_strategy="epoch",
save_strategy="epoch",
load_best_model_at_end=True,
)


# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
print("Starting training...")
trainer.train()
# Evaluate
print("Evaluating...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
# Save model
trainer.save_model('./sentiment_model')
print("Model saved!")


def predict_sentiment(text):
    """Predict sentiment for new text"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    confidence = torch.max(predictions).item()
    predicted_class = torch.argmax(predictions).item()
    sentiment = "Positive" if predicted_class == 1 else "Negative"

    return {
    'sentiment': sentiment,
    'confidence': confidence,
    'probabilities': {
        'negative': predictions[0][0].item(),
        'positive': predictions[0][1].item()
        }
    }

# Test with new examples
test_texts = [
"This movie is absolutely amazing!",
"I hate this boring film.",
"It's an okay movie, nothing special.",
"Fantastic storyline and great acting.",
"Completely terrible and disappointing."
]

print("\n" + "="*50)
print("TESTING TRAINED MODEL")
print("="*50)

for text in test_texts:
    result = predict_sentiment(text)
    print(f"\nText: {text}")
    print(f"Predicted: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Probabilities: Neg={result['probabilities']['negative']:.3f}, "
        f"Pos={result['probabilities']['positive']:.3f}")
    
# Simple Interface
import gradio as gr
def sentiment_interface(text):
    """Gradio interface for sentiment analysis"""
    result = predict_sentiment(text)
    return f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.3f}"
# Create interface
demo = gr.Interface(
    fn=sentiment_interface,
    inputs=gr.Textbox(placeholder="Enter text to analyze..."),
    outputs="text",
    title="Sentiment Analysis",
    description="Enter text to analyze its sentiment"
)
# Launch
demo.launch()