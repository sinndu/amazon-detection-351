import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# 1. Setup Data
# Mapping your labels: Human -> 0, AI -> 1
try:
    df = pd.read_csv("data/processed/analyzed_reviews_perplexity_burstiness.csv")
    print(f"Loaded {len(df)} reviews.")
except FileNotFoundError:
    print("Error: Could not find file")
    exit()

df['target'] = df['label'].map({'AI': 1, 'Human': 0})
X = df[['perplexity', 'burstiness']]
y = df['target']

# 20% test split as defined in your methodology 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Train Random Forest
# Using 100 trees as a standard baseline
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# 3. Evaluate
y_pred = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

# 4. Plotting Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot A: Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix: Random Forest')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')
ax1.set_xticklabels(['Human', 'AI'])
ax1.set_yticklabels(['Human', 'AI'])

# Plot B: Decision Boundary (The "Non-Linear" visualization)
DecisionBoundaryDisplay.from_estimator(
    rf_model, X.values, response_method="predict",
    cmap='RdBu', alpha=0.3, ax=ax2
)
sns.scatterplot(data=df, x='perplexity', y='burstiness', hue='label', 
                palette={'Human': 'blue', 'AI': 'red'}, s=20, alpha=0.6, ax=ax2)
ax2.set_title('Random Forest Decision Boundaries')
plt.xscale('log')
plt.tight_layout()
plt.show()