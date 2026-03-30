import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

INPUT_FILE = "data/processed/analyzed_reviews_perplexity_burstiness.csv"
OUTPUT_IMAGE = "model_combined_decision.png"

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} reviews.")
except FileNotFoundError:
    print(f"Error: Could not find '{INPUT_FILE}'")
    exit()

df['label_num'] = df['label'].apply(lambda x: 1 if x == 'AI' else 0)

X = df[['perplexity', 'burstiness']] # combine both features
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # test train

print("Training Combined Model (Perplexity + Burstiness)...")
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print(f"\nCombined Model Accuracy: {acc:.2%}")
print(classification_report(y_test, predictions, target_names=['Human', 'AI']))

plt.style.use('seaborn-v0_8')
plt.figure(figsize=(10, 8))

sns.scatterplot(
    data=df.iloc[y_test.index], 
    x='perplexity', 
    y='burstiness', 
    hue='label', 
    palette={'Human': 'blue', 'AI': 'red'},
    alpha=0.6,
    s=50
)

b = model.intercept_[0] # draw seperation line
w1, w2 = model.coef_.T

c = -b/w2
m = -w1/w2

xmin, xmax = 0, 150
ymin, ymax = 0, 50
xd = [xmin, xmax]
yd = [m*xmin + c, m*xmax + c]

plt.plot(xd, yd, 'k', lw=2, linestyle='--', label='Decision Boundary')

plt.title(f'AI vs Human Separation (Acc: {acc:.1%})')
plt.xlabel('Perplexity (Word Surprise)')
plt.ylabel('Burstiness (Rhythm Variance)')
plt.xlim(0, 150)
plt.ylim(0, 50)
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE)
print(f"Graph saved to '{OUTPUT_IMAGE}'")