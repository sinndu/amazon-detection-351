import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n-Random Forest Results")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
plt.title(f'Confusion Matrix: Random Forest\nAccuracy: {acc:.2%} | AUC: {auc:.2f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('cm_rf.png')
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(
    rf_model, X.values, response_method="predict",
    cmap='RdBu', alpha=0.3, ax=ax
)
sns.scatterplot(data=df, x='perplexity', y='burstiness', hue='label', 
                palette={'Human': 'blue', 'AI': 'red'}, s=20, alpha=0.6, ax=ax)
ax.set_title(f'Random Forest Decision Boundary (Acc: {acc:.1%})')
plt.xscale('log')
plt.tight_layout()
plt.savefig('boundary_rf.png')
plt.close()

importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
forest_importances = pd.Series(importances, index=X.columns)

fig, ax = plt.subplots(figsize=(8, 6))
forest_importances.plot.bar(yerr=std, ax=ax, color='seagreen', alpha=0.8)
ax.set_title("Random Forest: Feature Importance")
ax.set_ylabel("Mean Decrease in Impurity")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

print("Generating OOB Error Curve.")
rf_oob = RandomForestClassifier(warm_start=True, oob_score=True, max_depth=10, random_state=42)
error_rate = []

min_estimators = 15
max_estimators = 100

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore") 
    for i in range(min_estimators, max_estimators + 1):
        rf_oob.set_params(n_estimators=i)
        rf_oob.fit(X_train, y_train)
        oob_error = 1 - rf_oob.oob_score_
        error_rate.append((i, oob_error))

trees, errors = zip(*error_rate)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(trees, errors, color='forestgreen', lw=2)
ax.set_xlim(min_estimators, max_estimators)
ax.set_xlabel("Number of Trees (n_estimators)")
ax.set_ylabel("Out-of-Bag (OOB) Error Rate")
ax.set_title("Random Forest Convergence (OOB Error Curve)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('rf_oob_curve.png')
plt.close()

print("Saved all images: 'cm_rf.png', 'boundary_rf.png', 'rf_feature_importance.png', 'rf_oob_curve.png'")