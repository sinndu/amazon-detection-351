import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.inspection import DecisionBoundaryDisplay

try:
    df = pd.read_csv("data/processed/analyzed_reviews_perplexity_burstiness.csv")
    print(f"Loaded {len(df)} reviews.")
except FileNotFoundError:
    print("Error: Could not find file")
    exit()

df['target'] = df['label'].map({'AI': 1, 'Human': 0})
X = df[['perplexity', 'burstiness']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

xgb_model = XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    eval_metric=['logloss'],
    random_state=42
)
xgb_model.fit(X_train_sub, y_train_sub, eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)], verbose=False)

y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nXGBoost Results")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
plt.title(f'Confusion Matrix: XGBoost\nAccuracy: {acc:.2%} | AUC: {auc:.2f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('cm_xgb.png')
plt.close() 

fig, ax = plt.subplots(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(
    xgb_model, X.values, response_method="predict",
    cmap='RdBu', alpha=0.3, ax=ax
)
sns.scatterplot(data=df, x='perplexity', y='burstiness', hue='label', 
                palette={'Human': 'blue', 'AI': 'red'}, s=20, alpha=0.6, ax=ax)
ax.set_xscale('log')
ax.set_title(f'XGBoost Decision Boundary (Acc: {acc:.1%})')
plt.tight_layout()
plt.savefig('boundary_xgb.png')
plt.close()

results = xgb_model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train Loss', color='blue')
ax.plot(x_axis, results['validation_1']['logloss'], label='Validation Loss', color='red')
ax.legend()
ax.set_ylabel('Log Loss')
ax.set_xlabel('Boosting Iterations (Epochs)')
ax.set_title('XGBoost Training Loss Curve')
plt.tight_layout()
plt.savefig('xgb_loss_curve.png')
plt.close()

importances = xgb_model.feature_importances_
xgb_importances = pd.Series(importances, index=X.columns)

fig, ax = plt.subplots(figsize=(8, 6))
xgb_importances.plot.bar(ax=ax, color='darkorange', alpha=0.8)
ax.set_title("XGBoost: Feature Importance")
ax.set_ylabel("Relative Importance")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('xgb_feature_importance.png')
plt.close()

print("Saved all images: 'cm_xgb.png', 'boundary_xgb.png', 'xgb_loss_curve.png', 'xgb_feature_importance.png'")