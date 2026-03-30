import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    df = pd.read_csv("data/processed/analyzed_reviews_local.csv")
    print(f"Loaded {len(df)} rows.")
except:
    print("CSV not found.")
    exit()

quantile_95 = df['perplexity'].quantile(0.95) 
df_clean = df[df['perplexity'] < quantile_95] # remove top 5% of confusing outliers

print(f"Removed outliers above score: {quantile_95:.2f}")

plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(
    data=df_clean, 
    x='perplexity', 
    hue='label', 
    kde=True, 
    palette={'Human': 'blue', 'AI': 'red'}, 
    bins=40,
    ax=ax[0]
)
ax[0].set_title('View A: Standard Scale (Outliers Removed)')
ax[0].set_xlabel('Perplexity')

sns.histplot(
    data=df_clean, 
    x='perplexity', 
    hue='label', 
    kde=True, 
    palette={'Human': 'blue', 'AI': 'red'}, 
    bins=40,
    log_scale=True, 
    ax=ax[1]
)
ax[1].set_title('View B: Logarithmic Scale')
ax[1].set_xlabel('Log(Perplexity)')

plt.tight_layout()
plt.savefig('fixed_hills_visualization.png')
print("Created 'fixed_hills_visualization.png'.")