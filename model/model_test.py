import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


plt.rcParams['figure.facecolor'] = 'white'  
plt.rcParams['axes.edgecolor'] = 'black'   
plt.rcParams['savefig.facecolor'] = 'white' 
plt.rcParams['axes.linewidth'] = 1  

# Paths
test_file_path = "path to test dataset"
model_path = "./fine_tuned_roberta"

# Categories
categories = ["Innovative Technologies", "Physical Attributes", "Medical / Injuries",
              "Psychology", "Tactics analysis", "Scouting / Finance", "Other"]

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

def predict(text):
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
    predicted_label = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_label].item()
    return predicted_label, confidence

def plot_and_save_results(predictions_df):
    unique_categories = sorted(list(set(predictions_df['actual_label'].unique()) | 
                                  set(predictions_df['predicted_label'].unique())))
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    label_to_id = {label: idx for idx, label in enumerate(unique_categories)}
    
    conf_matrix = confusion_matrix(
        [label_to_id[label] for label in predictions_df['actual_label']],
        [label_to_id[label] for label in predictions_df['predicted_label']]
    )
    
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(len(unique_categories)), [c[:15] for c in unique_categories], rotation=45, ha='right')
    plt.yticks(range(len(unique_categories)), [c[:15] for c in unique_categories])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add numbers to confusion matrix
    for i in range(len(unique_categories)):
        for j in range(len(unique_categories)):
            plt.text(j, i, str(conf_matrix[i, j]),
                     ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Confidence Distribution
    plt.figure(figsize=(10, 8))
    correct_mask = predictions_df['actual_label'] == predictions_df['predicted_label']
    
    plt.hist([
        predictions_df[~correct_mask]['confidence'],
        predictions_df[correct_mask]['confidence']
    ], label=['Incorrect', 'Correct'], bins=20, alpha=0.7)
    
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Category-wise Accuracy
    plt.figure(figsize=(10, 8))
    category_accuracy = {}
    for category in unique_categories:
        mask = predictions_df['actual_label'] == category
        if mask.sum() > 0:
            correct = (predictions_df[mask]['actual_label'] == 
                    predictions_df[mask]['predicted_label']).mean()
            category_accuracy[category] = correct

    categories_present = list(category_accuracy.keys())
    accuracies = list(category_accuracy.values())

    plt.bar(range(len(categories_present)), accuracies, color='skyblue')
    plt.xticks(range(len(categories_present)), 
            [c[:15] for c in categories_present], 
            rotation=45, 
            ha='right')
    plt.title('Category-wise Accuracy')
    plt.xlabel('Category')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('category_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Average Confidence per Category
    plt.figure(figsize=(10, 8))
    avg_confidence = predictions_df.groupby('predicted_label')['confidence'].mean()
    plt.bar(range(len(avg_confidence)), avg_confidence, color='skyblue')
    plt.xticks(range(len(avg_confidence)), 
               [c[:15] for c in avg_confidence.index], 
               rotation=45, 
               ha='right')
    plt.title('Average Confidence per Category')
    plt.xlabel('Category')
    plt.ylabel('Average Confidence')
    
    plt.tight_layout()
    plt.savefig('average_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Load and prepare test data
    df_test = pd.read_csv(test_file_path, sep=',')
    df_test['text'] = (
        "Title: " + df_test['title'] +
        " Abstract: " + df_test['abstract'] +
        " Keywords: " + df_test['keywords'].fillna('')
    )

    # Filter labeled data points
    df_test = df_test[df_test['category'].notna()]
    df_test['labels'] = df_test['category'].apply(lambda x: categories.index(x.strip()))

    # Make predictions
    predictions = []
    for i, row in df_test.iterrows():
        pred_label, confidence = predict(row['text'])
        predictions.append({
            "text": row['text'],
            "actual_label": categories[row['labels']],
            "predicted_label": categories[pred_label],
            "confidence": confidence
        })

    predictions_df = pd.DataFrame(predictions)

    plot_and_save_results(predictions_df)
    
    print("\nClassification Report:")
    print(classification_report(
        predictions_df['actual_label'],
        predictions_df['predicted_label']
    ))
    
    accuracy = (predictions_df['actual_label'] == predictions_df['predicted_label']).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
