import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

INPUT_FILE = "data/processed/analyzed_reviews_perplexity_burstiness.csv"
OUTPUT_IMAGE = "burstiness_hills.png"

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: Could not find '{INPUT_FILE}'.")
    exit()

quantile_99 = df['burstiness'].quantile(0.99) # clip outliers (too long reviews)
df_clean = df[df['burstiness'] < quantile_99]

print(f"Plotting data (clipped outliers above {quantile_99:.2f})...")

plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(
    data=df_clean, 
    x='burstiness', 
    hue='label', 
    kde=True, 
    palette={'Human': 'blue', 'AI': 'red'}, 
    bins=40,
    alpha=0.6,
    ax=ax[0]
)
ax[0].set_title('View A: Burstiness (Standard Scale)')
ax[0].set_xlabel('Burstiness Score (Std Dev of Sentence Lengths)')
ax[0].set_ylabel('Count')

df_log_safe = df_clean[df_clean['burstiness'] > 0]

sns.histplot(
    data=df_log_safe, 
    x='burstiness', 
    hue='label', 
    kde=True, 
    palette={'Human': 'blue', 'AI': 'red'}, 
    bins=40,
    log_scale=True,  
    alpha=0.6,
    ax=ax[1]
)
ax[1].set_title('View B: Log-Burstiness (No Zeros)')
ax[1].set_xlabel('Log(Burstiness)')

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE)
print(f"Graph saved to '{OUTPUT_IMAGE}'")