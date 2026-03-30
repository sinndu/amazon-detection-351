import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

INPUT_FILE = "data/processed/analyzed_reviews_perplexity_burstiness.csv"
OUTPUT_IMAGE = "model_burstiness_only.png"

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} reviews.")
except FileNotFoundError:
    print(f"Error: Could not find '{INPUT_FILE}'")
    exit()

df['label_num'] = df['label'].apply(lambda x: 1 if x == 'AI' else 0)

X = df[['burstiness']] 
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # test and train

print("Training Burstiness-Only Model...")
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test) # evaluate
acc = accuracy_score(y_test, predictions)

print(f"\nBurstiness Accuracy: {acc:.2%}")
print(classification_report(y_test, predictions, target_names=['Human', 'AI']))
