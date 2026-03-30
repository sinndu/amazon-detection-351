import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.multivariate.manova import MANOVA

df = pd.read_csv('data/processed/analyzed_reviews_perplexity_burstiness.csv') 

human = df[df['label'] == 'Human']
ai = df[df['label'] == 'AI'] 

t_stat_p, p_val_t_p = stats.ttest_ind(human['perplexity'], ai['perplexity'], equal_var=False)
u_stat_p, p_val_u_p = stats.mannwhitneyu(human['perplexity'], ai['perplexity'], alternative='two-sided')

print("\n[Perplexity Differences]")
print(f"Welch's T-test p-value:   {p_val_t_p:.4e}")
print(f"Mann-Whitney U p-value:   {p_val_u_p:.4e}")

t_stat_b, p_val_t_b = stats.ttest_ind(human['burstiness'], ai['burstiness'], equal_var=False)
u_stat_b, p_val_u_b = stats.mannwhitneyu(human['burstiness'], ai['burstiness'], alternative='two-sided')

print("\n[Burstiness Differences]")
print(f"Welch's T-test p-value:   {p_val_t_b:.4e}")
print(f"Mann-Whitney U p-value:   {p_val_u_b:.4e}")

print("\n[Combined Features - MANOVA]")
manova = MANOVA.from_formula('perplexity + burstiness ~ label', data=df)
print(manova.mv_test())