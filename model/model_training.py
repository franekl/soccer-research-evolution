import pandas as pd
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

# Combine datasets
df = pd.read_csv('path to training data')

# Clean and preprocess combined dataset
df['category'] = df['category'].replace(r" \\/ ", " / ", regex=True)
categories = ["Innovative Technologies", "Physical Attributes", "Medical / Injuries",
              "Psychology", "Tactics analysis", "Scouting / Finance", "Other", "Not Soccer Related"]

df['labels'] = df['category'].apply(lambda x: categories.index(x.strip()))
df['text'] = (
    "Title: " + df['title'] + 
    " Abstract: " + df['abstract'] + 
    " Keywords: " + df['keywords'].fillna('')
)


# Check class distribution
print("\nClass distribution before balancing:")
print(df['labels'].value_counts())

# Data balancing: Upsample minority classes
max_samples = df['labels'].value_counts().max()
df_balanced = df.groupby('labels').apply(
    lambda x: x.sample(n=max_samples, replace=True, random_state=42)
).reset_index(drop=True)

# New class distribution
print("\nClass distribution after balancing:")
print(df_balanced['labels'].value_counts())

# Stratified split
train_df, eval_df = train_test_split(
    df_balanced, test_size=0.2, stratify=df_balanced['labels'], random_state=42
)

# Create datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Load tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Format datasets
for dataset in [train_dataset, eval_dataset]:
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load Roberta model
model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(categories)
)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,  # Standard Transformers learning rate
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    save_total_limit=2,
    seed=42
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted")
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./fine_tuned_roberta")
tokenizer.save_pretrained("./fine_tuned_roberta")

# Prediction function
def predict(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_label].item()
    
    return {
        "category": categories[predicted_label],
        "confidence": confidence
    }
