import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# 1. Setup Data
# Mapping your labels: Human -> 0, AI -> 1
df = pd.read_csv("data/processed/analyzed_reviews_perplexity_burstiness.csv")
print(f"Loaded {len(df)} reviews.")
df['target'] = df['label'].map({'AI': 1, 'Human': 0})
X = df[['perplexity', 'burstiness']]
y = df['target']

# Maintaining the 20% test split from your midterm methodology [cite: 71]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Train XGBoost
# We use logloss to evaluate performance during training
xgb_model = XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 3. Evaluate
y_pred = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

# 4. Plotting with Log Scale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds', ax=ax1)
ax1.set_title('Confusion Matrix: XGBoost')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')
ax1.set_xticklabels(['Human', 'AI'])
ax1.set_yticklabels(['Human', 'AI'])

# Decision Boundary with Log Scale Fix
# We plot the boundary on the standard scale but set the axis to log for viewing
DecisionBoundaryDisplay.from_estimator(
    xgb_model, X.values, response_method="predict",
    cmap='RdBu', alpha=0.3, ax=ax2
)
sns.scatterplot(data=df, x='perplexity', y='burstiness', hue='label', 
                palette={'Human': 'blue', 'AI': 'red'}, s=20, alpha=0.6, ax=ax2)

ax2.set_xscale('log') # This fixes the "squashed at zero" look
ax2.set_title('XGBoost Decision Boundaries (Log Scale)')
plt.tight_layout()
plt.show()