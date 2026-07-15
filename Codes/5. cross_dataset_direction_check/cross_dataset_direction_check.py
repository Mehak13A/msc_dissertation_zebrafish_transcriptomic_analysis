#!/usr/bin/env python3
"""
5. Cross-Dataset Direction Check

PURPOSE
    Ask whether the age-dependent telomerase phenotype shows any early directional signal. The 5,541 genes that are
    significantly dysregulated at 18 months are located in the 3-month data, and their 3-month fold changes are examined:
        - If the phenotype is genuinely age-dependent, those genes should sit centred on zero at 3 months (no early drift).
        - If there is early drift, the same genes should already be skewed at 3 months in the same direction they take at 18 months.

    The check is directional and distributional (means, one-sample t-tests, a non-parametric comparison against the
    genome-wide background, and a direct 18mo-vs-3mo fold-change correlation), not a re-run of differential expression.

INPUTS
    Two differential-expression result tables produced by earlier programs in this archive (gene IDs as the index; a 'log2FoldChange' column; a significance column):

    18-month Q1 genotype results  : output of differential_expression_analysis.py (q1_genotype_DEG_results.csv).
                                    A 'qvalue' column is preferred; 'padj' is used as a fallback.
    3-month MUT-vs-WT results     : output of tert_3month_de_analysis.py (tert3m_MUT_vs_WT_results.csv).

OUTPUTS (all written to OUTPUT_DIR)
    cross_dataset_density.png            : KDE of 3mo log2FC for each gene set.
    cross_dataset_boxplot.png            : boxplot of 3mo log2FC by gene set.
    cross_dataset_scatter.png            : 18mo vs 3mo log2FC for shared DEGs (skipped if no shared DEGs exist).
    cross_dataset_cdf.png                : cumulative distribution of 3mo log2FC.
    cross_dataset_direction_summary.csv  : per-gene 3mo/18mo fold changes, 18mo significance and direction.

References (this file only)
    These references apply to this source file only and are independent of any reference numbering used in the accompanying report.

    [1] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python", Nat. Methods,
        vol. 17, pp. 261-272, 2020.
    [2] M. L. Waskom, "seaborn: statistical data visualization", J. Open Source Software, vol. 6, no. 60, art. 3021, 2021.
"""

# Import all necessary libraries and modules

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # non-interactive backend: render figures to file
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION (Input and output paths as per my computer)

# Q1 (18-month genotype) results — output of differential_expression_analysis.py
Q1_18M_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/1. differential_expression_analysis/q1_genotype_DEG_results.csv'
# 3-month MUT vs WT results — output of tert_3month_de_analysis.py
Q1_3M_CSV  = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/4. tert_3month_de_analysis/tert3m_MUT_vs_WT_results.csv'
# Directory where this script's figures and summary CSV are written
OUTPUT_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/5. cross_dataset_direction_check'
# Create the output directory if it does not already exist (no error if present)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Cross-Dataset Direction Check: 18mo DEGs in 3mo data")

# 1. Load both result tables
# Gene IDs are the row index in both tables so the two datasets can be aligned.
results_18mo = pd.read_csv(Q1_18M_CSV, index_col=0)
results_3mo  = pd.read_csv(Q1_3M_CSV,  index_col=0)

# Prefer Storey q-values for the 18-month significance call; fall back to the DESeq2 'padj' (BH) column if the q-value
# column is absent, and warning added so the upstream analysis can be re-run to add it.
significance_column = 'qvalue' if 'qvalue' in results_18mo.columns else 'padj'
if significance_column == 'padj':
    print("WARNING: using padj for 18mo - re-run differential_expression_analysis.py to get qvalue column")

# 2. Get the 18-month DEG sets
# Significant at 18 months: q (or padj) < 0.05 AND |log2FC| > 1, matching the main pipeline's significance definition.
degs_18mo = results_18mo[(results_18mo[significance_column] < 0.05) & (results_18mo['log2FoldChange'].abs() > 1)]
down_in_mut_18mo = degs_18mo[degs_18mo['log2FoldChange'] < 0] # down in MUT at 18mo
up_in_mut_18mo   = degs_18mo[degs_18mo['log2FoldChange'] > 0] # up in MUT at 18mo

print(f"\n18-month DEGs loaded: {len(degs_18mo)} total")
print(f"  Down in MUT (18mo): {len(down_in_mut_18mo)}")
print(f"  Up in MUT (18mo):   {len(up_in_mut_18mo)}")
print(f"\n3-month genes loaded: {len(results_3mo)}")

# 3. Find shared gene IDs
# Only genes present in BOTH tables can be compared across ages.
shared_gene_ids = results_3mo.index.intersection(results_18mo.index)
shared_down_ids = results_3mo.index.intersection(down_in_mut_18mo.index)
shared_up_ids = results_3mo.index.intersection(up_in_mut_18mo.index)

print(f"\nShared gene IDs (all):        {len(shared_gene_ids)}")
print(f"18mo down-in-MUT genes found in 3mo data: {len(shared_down_ids)}")
print(f"18mo up-in-MUT genes found in 3mo data:   {len(shared_up_ids)}")

# 4. Extract 3-month log2FC for these gene sets
# For each 18-month gene set, pull the corresponding 3-month fold changes. The full shared set acts as the genome-wide background for comparison.
lfc_3mo_down_set = results_3mo.loc[shared_down_ids, 'log2FoldChange'].dropna()
lfc_3mo_up_set = results_3mo.loc[shared_up_ids, 'log2FoldChange'].dropna()
lfc_3mo_background = results_3mo.loc[shared_gene_ids, 'log2FoldChange'].dropna()

print("\n3mo log2FC values available:")
print(f"  Background (all shared genes): {len(lfc_3mo_background)}")
print(f"  18mo down-in-MUT set:          {len(lfc_3mo_down_set)}")
print(f"  18mo up-in-MUT set:            {len(lfc_3mo_up_set)}")

# 5. Statistical tests
# One-sample t-test: is each set's mean 3mo log2FC significantly different from zero? (A mean near zero == no early drift.)
t_stat_down, p_value_down = stats.ttest_1samp(lfc_3mo_down_set, 0)
t_stat_up, p_value_up   = stats.ttest_1samp(lfc_3mo_up_set, 0)

# Mann-Whitney U (non-parametric): does each set's 3mo distribution differ from the genome-wide background, without assuming normality? [1]
mannwhitney_down = stats.mannwhitneyu(lfc_3mo_down_set, lfc_3mo_background, alternative='two-sided')
mannwhitney_up   = stats.mannwhitneyu(lfc_3mo_up_set,   lfc_3mo_background, alternative='two-sided')

print("\nStatistical tests")
print("18mo DOWN genes at 3mo:")
print(f"  Mean log2FC = {lfc_3mo_down_set.mean():+.4f}  (0 = no early drift)")
print(f"  One-sample t-test vs 0:  t={t_stat_down:.3f}, p={p_value_down:.4f}")
print(f"  Mann-Whitney vs background: p={mannwhitney_down.pvalue:.4f}")

print("\n18mo UP genes at 3mo:")
print(f"  Mean log2FC = {lfc_3mo_up_set.mean():+.4f}  (0 = no early drift)")
print(f"  One-sample t-test vs 0:  t={t_stat_up:.3f}, p={p_value_up:.4f}")
print(f"  Mann-Whitney vs background: p={mannwhitney_up.pvalue:.4f}")

# 6. FIGURE 1: Density plot - distribution of 3mo log2FC for each gene set
# Overlaid kernel-density estimates let us see whether the up/down sets are shifted away from the background (which is centred on zero).
fig, ax = plt.subplots(figsize=(9, 5))

lfc_3mo_background.plot.kde(ax=ax, color='#AAAAAA', linewidth=1.5, label=f'All shared genes (n={len(lfc_3mo_background):,})', linestyle='--')
lfc_3mo_down_set.plot.kde(ax=ax, color='#457B9D', linewidth=2, label=f'18mo down-in-MUT (n={len(lfc_3mo_down_set):,})\n' f'mean={lfc_3mo_down_set.mean():+.3f}')
lfc_3mo_up_set.plot.kde(ax=ax, color='#E63946', linewidth=2, label=f'18mo up-in-MUT (n={len(lfc_3mo_up_set):,})\n' f'mean={lfc_3mo_up_set.mean():+.3f}')

ax.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5) # zero reference
ax.set_xlim(-6, 6)
ax.set_xlabel('log2 Fold Change at 3 months (MUT vs WT)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('18-month DEGs: distribution of fold change at 3 months', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.2)
plt.tight_layout()
density_fig_path = os.path.join(OUTPUT_DIR, 'cross_dataset_density.png')
plt.savefig(density_fig_path, dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved: cross_dataset_density.png")

# 7. FIGURE 2: Boxplot comparison
# The same three groups as boxplots, with each group's mean annotated, so the central tendency and spread can be read directly.
fig, ax = plt.subplots(figsize=(7, 5))

# Stack the three sets into one long-form frame with a 'group' label for seaborn.
boxplot_data = pd.DataFrame({
    'log2FC_3mo': pd.concat([lfc_3mo_background, lfc_3mo_down_set, lfc_3mo_up_set]),
    'group': (['Background'] * len(lfc_3mo_background) + ['18mo Down\nin MUT'] * len(lfc_3mo_down_set) + ['18mo Up\nin MUT'] * len(lfc_3mo_up_set))
})

box_colors = {'Background': '#CCCCCC', '18mo Down\nin MUT': '#457B9D', '18mo Up\nin MUT': '#E63946'}
group_order = ['Background', '18mo Down\nin MUT', '18mo Up\nin MUT']

sns.boxplot(data=boxplot_data, x='group', y='log2FC_3mo', order=group_order, palette=box_colors, width=0.5, fliersize=2, ax=ax)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6) # zero reference

# Annotate each box with its group mean, placed near the top of the axes.
for i, grp in enumerate(group_order):
    group_mean = boxplot_data[boxplot_data['group'] == grp]['log2FC_3mo'].mean()
    ax.text(i, ax.get_ylim()[1] * 0.92, f'mean={group_mean:+.3f}', ha='center', fontsize=9, color='black')

ax.set_xlabel('')
ax.set_ylabel('log2 Fold Change at 3 months (MUT vs WT)', fontsize=11)
ax.set_title('3-month signal in genes that become DEGs at 18 months', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
boxplot_fig_path = os.path.join(OUTPUT_DIR, 'cross_dataset_boxplot.png')
plt.savefig(boxplot_fig_path, dpi=200, bbox_inches='tight')
plt.close()
print("Saved: cross_dataset_boxplot.png")

# 8. FIGURE 3: Scatter - 18mo log2FC vs 3mo log2FC for all shared DEGs
# Gene-by-gene, does the 18-month fold change predict the 3-month fold change? A near-zero correlation supports age-dependent onset.
# Skipped if the two datasets share no significant DEGs (the expected age-dependent result).
shared_deg_ids = results_3mo.index.intersection(degs_18mo.index)
if len(shared_deg_ids) > 0:
    fc_18mo_shared = results_18mo.loc[shared_deg_ids, 'log2FoldChange']
    fc_3mo_shared  = results_3mo.loc[shared_deg_ids,  'log2FoldChange'].reindex(shared_deg_ids)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(fc_18mo_shared, fc_3mo_shared, s=15, alpha=0.5, color='#555555', edgecolors='none')
    ax.axhline(0, color='grey', linewidth=0.7, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.7, linestyle='--')

    # Pearson correlation of the paired fold changes (drop any NaN pairs first). [1]
    valid_pairs = pd.DataFrame({'x': fc_18mo_shared, 'y': fc_3mo_shared}).dropna()
    if len(valid_pairs) > 5:
        correlation_r, correlation_p = stats.pearsonr(valid_pairs['x'], valid_pairs['y'])
        ax.text(0.05, 0.95, f'r = {correlation_r:.3f}\nn = {len(valid_pairs)}', transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel('log2FC at 18 months', fontsize=12)
    ax.set_ylabel('log2FC at 3 months', fontsize=12)
    ax.set_title('18-month DEGs: fold change at 18mo vs 3mo', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    scatter_fig_path = os.path.join(OUTPUT_DIR, 'cross_dataset_scatter.png')
    plt.savefig(scatter_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: cross_dataset_scatter.png")
else:
    print("No shared DEGs between 3mo and 18mo - scatter plot skipped (expected result).")

# 9. FIGURE 4: Cumulative distribution (CDF)
# Empirical CDFs make a small directional shift easy to see: a set drifting negative sits to the LEFT of the background,
# positive drift to the RIGHT.
fig, ax = plt.subplots(figsize=(8, 5))

for lfc_values, label, color, line_width in [
    (lfc_3mo_background, f'Background (n={len(lfc_3mo_background):,})', '#AAAAAA', 1.5),
    (lfc_3mo_down_set,   f'18mo Down-in-MUT (n={len(lfc_3mo_down_set):,})', '#457B9D', 2.2),
    (lfc_3mo_up_set,     f'18mo Up-in-MUT (n={len(lfc_3mo_up_set):,})',   '#E63946', 2.2),
]:
    sorted_values = np.sort(lfc_values) # x-axis: sorted log2FC
    cumulative_proportion = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax.plot(sorted_values, cumulative_proportion, label=label, color=color, linewidth=line_width)

ax.axvline(0, color='black', linewidth=0.8, alpha=0.4) # zero reference
ax.set_xlim(-4, 4)
ax.set_xlabel('log2 Fold Change at 3 months (MUT vs WT)', fontsize=12)
ax.set_ylabel('Cumulative proportion', fontsize=12)
ax.set_title('CDF: do 18-month DEGs show early directional bias at 3 months?', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
cdf_fig_path = os.path.join(OUTPUT_DIR, 'cross_dataset_cdf.png')
plt.savefig(cdf_fig_path, dpi=200, bbox_inches='tight')
plt.close()
print("Saved: cross_dataset_cdf.png")

# 10. Save summary CSV
# One row per shared gene: its 3mo and 18mo fold changes, the 18mo significance value, and its 18mo direction (up/down in MUT, or not significant).
summary_rows = []
for gene_id in shared_gene_ids:
    fc_3mo  = results_3mo.loc[gene_id, 'log2FoldChange'] if gene_id in results_3mo.index else np.nan
    fc_18mo = results_18mo.loc[gene_id, 'log2FoldChange'] if gene_id in results_18mo.index else np.nan
    q_18mo  = results_18mo.loc[gene_id, significance_column] if gene_id in results_18mo.index else np.nan
    is_deg_18mo = gene_id in degs_18mo.index
    if is_deg_18mo:
        direction_18mo = 'up_in_MUT' if fc_18mo > 0 else 'down_in_MUT'
    else:
        direction_18mo = 'not_significant'
    summary_rows.append({
        'gene_id': gene_id,
        'gene_name': results_18mo.loc[gene_id, 'gene_name'] if ('gene_name' in results_18mo.columns and gene_id in results_18mo.index) else '',
        'log2FC_3mo': round(fc_3mo, 4) if pd.notna(fc_3mo) else np.nan,
        'log2FC_18mo': round(fc_18mo, 4) if pd.notna(fc_18mo) else np.nan,
        f'{significance_column}_18mo': round(q_18mo, 6) if pd.notna(q_18mo) else np.nan,
        'sig_at_18mo': is_deg_18mo,
        'direction_18mo': direction_18mo,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'cross_dataset_direction_summary.csv'), index=False)
print("Saved: cross_dataset_direction_summary.csv")

# 11. Print interpretation
# Plain-language read-out of the two key means and t-tests, so the result can be interpreted straight from the console without opening the figures.
print("\n")
print("INTERPRETATION")
print(f"""
Key question: Do the 18-month DEGs show early directional signal at 3 months?

18mo DOWN-in-MUT genes at 3mo:
  Mean log2FC = {lfc_3mo_down_set.mean():+.4f}
  -> If close to 0: no early drift (age-dependent onset confirmed)
  -> If negative:   subtle early suppression already present

18mo UP-in-MUT genes at 3mo:
  Mean log2FC = {lfc_3mo_up_set.mean():+.4f}
  -> If close to 0: no early drift (age-dependent onset confirmed)
  -> If positive:   subtle early induction already present

One-sample t-test (vs 0):
  Down genes: p = {p_value_down:.4f}  {' early drift detected ' if p_value_down < 0.05 else '(not significant - centred on zero)'}
  Up genes:   p = {p_value_up:.4f}  {' early drift detected ' if p_value_up < 0.05 else '(not significant - centred on zero)'}
""")
print("Process finished successfully!")