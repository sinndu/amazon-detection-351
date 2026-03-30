import pandas as pd
import torch
import math
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

HUMAN_PATH = r"data\collected\collected-llama-3.1-8b-instant-20260213-1940\human_reviews.csv"
AI_PATH = r"data\collected\collected-llama-3.1-8b-instant-20260213-1940\synthetic_reviews.csv"
OUTPUT_IMAGE = "perplexity_hills.png"

print("Loading GPT-2 Model.")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

def calculate_perplexity(text):
    if not isinstance(text, str) or len(text) < 10:
        return None

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    if input_ids.shape[1] > 1024: # GPT-2 can handle 1024 at once
        input_ids = input_ids[:, :1024]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
        
    return perplexity

print("Loading CSVs:")
try:
    df_human = pd.read_csv(HUMAN_PATH)
    df_human['label'] = 'Human'
    
    df_ai = pd.read_csv(AI_PATH)
    df_ai['label'] = 'AI'
    
    df = pd.concat([df_human, df_ai], ignore_index=True)
    print(f"Loaded {len(df)} total reviews.")
except FileNotFoundError:
    print(f"Error: Could not find files at {HUMAN_PATH}")
    exit()

print("Calculating Perplexity...")
tqdm.pandas()
df['perplexity'] = df['text'].progress_apply(calculate_perplexity)

df = df.dropna(subset=['perplexity']) # drop failed rows (above 1024 tokens likely)

df.to_csv("outputs/analyzed_reviews_local.csv", index=False)
print("Saved data to analyzed_reviews_local.csv")
