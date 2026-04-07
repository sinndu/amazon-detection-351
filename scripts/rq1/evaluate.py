import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.multivariate.manova import MANOVA

# Load your analyzed dataset
# Adjust path if necessary
df = pd.read_csv('data/processed/analyzed_reviews_perplexity_burstiness.csv') 

human = df[df['label'] == 'Human']
ai = df[df['label'] == 'AI'] 

def calculate_cohens_d(group1, group2):
    """Calculates Cohen's d for effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def calculate_rank_biserial(u_stat, n1, n2):
    """Calculates Rank-Biserial Correlation (r) for Mann-Whitney U."""
    return 1 - (2 * u_stat) / (n1 * n2)

print("--- DESCRIPTIVE STATISTICS ---")
for feat in ['perplexity', 'burstiness']:
    print(f"\n[{feat.capitalize()}]")
    print(f"Human: Mean = {human[feat].mean():.2f}, Std Dev = {human[feat].std():.2f}")
    print(f"AI:    Mean = {ai[feat].mean():.2f}, Std Dev = {ai[feat].std():.2f}")

print("\n--- INFERENTIAL STATISTICS & EFFECT SIZES ---")

# --- Perplexity ---
t_stat_p, p_val_t_p = stats.ttest_ind(human['perplexity'], ai['perplexity'], equal_var=False)
u_stat_p, p_val_u_p = stats.mannwhitneyu(human['perplexity'], ai['perplexity'], alternative='two-sided')
d_p = calculate_cohens_d(human['perplexity'], ai['perplexity'])
r_p = calculate_rank_biserial(u_stat_p, len(human), len(ai))

print("\n[Perplexity]")
print(f"Welch's T-test: t = {t_stat_p:.4f}, p = {p_val_t_p:.4e}, Cohen's d = {d_p:.4f}")
print(f"Mann-Whitney U: U = {u_stat_p}, p = {p_val_u_p:.4e}, r = {r_p:.4f}")

# --- Burstiness ---
t_stat_b, p_val_t_b = stats.ttest_ind(human['burstiness'], ai['burstiness'], equal_var=False)
u_stat_b, p_val_u_b = stats.mannwhitneyu(human['burstiness'], ai['burstiness'], alternative='two-sided')
d_b = calculate_cohens_d(human['burstiness'], ai['burstiness'])
r_b = calculate_rank_biserial(u_stat_b, len(human), len(ai))

print("\n[Burstiness]")
print(f"Welch's T-test: t = {t_stat_b:.4f}, p = {p_val_t_b:.4e}, Cohen's d = {d_b:.4f}")
print(f"Mann-Whitney U: U = {u_stat_b}, p = {p_val_u_b:.4e}, r = {r_b:.4f}")

print("\n--- MULTIVARIATE ANALYSIS (MANOVA) ---")
manova = MANOVA.from_formula('perplexity + burstiness ~ label', data=df)
mv_results = manova.mv_test()
print(mv_results)

# Extra breakdown for the report:
wilks = mv_results.results['label']['stat'].iloc[0, 0]
print(f"\nWilks' Lambda for Label: {wilks:.4f}")
print(f"Explained Variance (1 - Lambda): {(1 - wilks)*100:.2f}%")