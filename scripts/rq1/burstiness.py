import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

INPUT_FILE = "data/processed/analyzed_reviews_perplexity.csv"
OUTPUT_FILE = "data/processed/analyzed_reviews_perplexity_burstiness.csv"

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} reviews.")
except FileNotFoundError:
    print(f"Error: Could not find '{INPUT_FILE}'.")
    exit()

def calculate_burstiness(text):
    if not isinstance(text, str) or len(text) < 10:
        return 0
    
    try:
        sentences = sent_tokenize(text)
    except Exception:
        return 0
        
    if len(sentences) <= 1:
        return 0 # no variation if only 1 sentence
    
    lengths = [len(s.split()) for s in sentences]
    
    return np.std(lengths)

print("Calculating Burstiness.")
tqdm.pandas()
df['burstiness'] = df['text'].progress_apply(calculate_burstiness)

df.to_csv(OUTPUT_FILE, index=False)
print(f"Success. Saved enhanced data to '{OUTPUT_FILE}'")