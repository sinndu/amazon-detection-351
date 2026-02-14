import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HUMAN_FILE = r'data\collected\collected-20260213-1940\human_reviews.csv'  
AI_FILE = r'data\collected\collected-20260213-1940\synthetic_reviews.csv'

def calculate_perplexity(text):
    if not isinstance(text, str): return 0
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": text}],
            max_tokens=1,     
            logprobs=True,    
            top_logprobs=1
        )
        
        if hasattr(completion.choices[0], 'logprobs') and completion.choices[0].logprobs:
            token_logprobs = [t.logprob for t in completion.choices[0].logprobs.content]
            if not token_logprobs: return 0
            
            avg_logprob = np.mean(token_logprobs)
            perplexity = np.exp(-avg_logprob)
            return perplexity
            
        return 0 

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            tqdm.write(f"Rate limit hit. Waiting 10s. (Error: {error_msg[:50]})")
            time.sleep(10)
        else:
            tqdm.write(f"API Error: {error_msg}")
            
        return None

df_human = pd.read_csv(HUMAN_FILE)
df_human['label'] = 'Human'

df_ai = pd.read_csv(AI_FILE)
df_ai['label'] = 'AI'

df = pd.concat([df_human, df_ai], ignore_index=True)
print(f"Loaded {len(df)} total reviews.")

tqdm.pandas()

def rate_limited_perplexity(text):
    time.sleep(2.1) 
    return calculate_perplexity(text)

print("Calculating Perplexity...")
df['perplexity'] = df['text'].progress_apply(rate_limited_perplexity)

df.to_csv('analyzed_reviews.csv', index=False)
print("Saved to analyzed_reviews.csv")

plt.style.use('seaborn-v0_8')
plt.figure(figsize=(10, 6))

df['perplexity_clipped'] = df['perplexity'].clip(upper=150) 

sns.histplot(data=df, x='perplexity_clipped', hue='label', kde=True, 
             palette={'Human': 'blue', 'AI': 'red'}, bins=30, alpha=0.6)

plt.title('Perplexity Distribution (Human vs AI)')
plt.xlabel('Perplexity Score')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('perplexity_hill.png')
print("Chart saved as perplexity_hill.png")