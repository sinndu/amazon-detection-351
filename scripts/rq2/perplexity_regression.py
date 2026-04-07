import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv("data/processed/analyzed_reviews_perplexity.csv")
except:
    print("CSV not found.")
    exit()

df['label_num'] = df['label'].apply(lambda x: 1 if x == 'AI' else 0)
X = df[['perplexity']] 
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Perplexity-Only Model...")
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1] # Get probabilities for AUC

accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probs)

print(f"\nModel Accuracy: {accuracy:.2%}")
print(f"ROC-AUC Score: {auc:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=['Human', 'AI']))

# Generate Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
plt.title(f'Confusion Matrix: Perplexity Only\nAccuracy: {accuracy:.2%} | AUC: {auc:.2f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('cm_perplexity_only.png')
print("Saved confusion matrix to 'cm_perplexity_only.png'")