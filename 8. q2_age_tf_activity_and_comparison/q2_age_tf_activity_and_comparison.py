#!/usr/bin/env python3
"""
8. Natural-Ageing TF Activity (Q2) and Full Q1 vs Q2 Comparison

PURPOSE
    Answers the question: "is telomerase loss just accelerated ageing?". It scores the FULL CollecTRI regulon collection (all ~1,185 transcription factors)
    against the Q2 age contrast (old 37mo versus young wild-type 11mo) from scratch, giving an independent ranked TF-activity list for natural ageing.
    It then compares that against the full Q1 (genotype) TF-activity table. Every TF is scored independently in each contrast;
    the 71-TF exported network is NOT used anywhere, so the comparison is fair in both directions.

    This is the same scoring method used to build the original Q1 table (decoupler univariate linear model on CollecTRI regulons), applied to Q2.

    REQUIRES AN INTERNET CONNECTION: decoupler downloads CollecTRI via OmniPath. Run this on a machine with internet (not needed for the orthology step, which reuses the saved map).

INPUTS
    AGE_DEG_CSV       : the Q2 age DE table (old 37mo versus young wild-type 11mo, e.g. q2_age_DEG_results.csv), indexed by
                        zebrafish gene ID with a 'stat' (preferred) and 'log2FoldChange' column.
    ORTHOLOGY_MAP_CSV : orthology_map_zfish_to_human.csv (output of grn_tf_activity_model.py), columns ['zfish_id','human_symbol'].
    TF_ACTIVITY_Q1_CSV: tf_activity_Q1.csv (output of grn_tf_activity_model.py), indexed by TF with 'activity' and 'pval'
                        columns. This is the full Q1 table used for the comparison, so the Q1 side matches the values already reported elsewhere in the project.

OUTPUTS (all written to OUTPUT_DIR)
    tf_activity_age.csv        : ranked TF activities and p-values for natural ageing (all scored TFs).
    q1_vs_age_tf_comparison.csv: per-TF Q1 activity, ageing activity, significance flags and direction agreement.
    q1_vs_age_scatter.png      : Q1 versus ageing TF activity, TFs significant in both contrasts labelled.

References (this file only)
    These references apply to this source file only and are independent of any reference numbering used in the accompanying report.

    [1] S. Müller-Dott et al., "Expanding the coverage of regulons from high-confidence prior knowledge for accurate
        estimation of transcription factor activities", Nucleic Acids Research, vol. 51, no. 20, pp. 10934-10949, 2023.
    [2] P. Badia-i-Mompel et al., "decoupleR: ensemble of computational methods to infer biological activities from
        omics data", Bioinformatics Advances, vol. 2, no. 1, art. vbac016, 2022.
    [3] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python", Nature Methods, vol. 17, pp. 261-272, 2020.
"""

# Import all necessary libraries and modules

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # non-interactive backend: render figures to files
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# CONFIGURATION (input and output paths as per my computer)

# Input and output paths
AGE_DEG_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/1. differential_expression_analysis/q2_age_DEG_results.csv'
ORTHOLOGY_MAP_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model/orthology_map_zfish_to_human.csv'
TF_ACTIVITY_Q1_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model/tf_activity_Q1.csv'
OUTPUT_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/8. q2_age_tf_activity_and_comparison'

STAT_COL = 'stat' # gene-level signal fed to decoupler ('stat' preferred; falls back to log2FoldChange)
LOG2FC_COL = 'log2FoldChange'
AGG = 'maxabs' # collapse many zebrafish genes onto one human symbol
SIG_TF_PVAL = 0.05
# Transcription factors to call out in the console (only shown if scored)
KEY_TFS = ['E2F1', 'E2F3', 'E2F4', 'TP53', 'AP1', 'JUN', 'NFKB', 'HIF1A', 'SMAD3', 'MYC']

def load_orthology_lookup(path):
    """Clean orthology table with an upper-case human_symbol column."""
    ortho = pd.read_csv(path).dropna(subset=['human_symbol'])
    ortho = ortho[~ortho['human_symbol'].isin(['N/A', 'nan', 'None', ''])].copy()
    ortho['human_symbol'] = ortho['human_symbol'].astype(str).str.upper()
    return ortho

def build_human_stat_vector(deg_path, ortho, stat_col, agg='maxabs'):
    """Map a zebrafish DE table onto human symbols, one statistic per symbol."""
    deg = pd.read_csv(deg_path, index_col=0)
    column = stat_col if stat_col in deg.columns else LOG2FC_COL
    stat_series = deg[column].dropna()
    merged = ortho.merge(stat_series.rename('stat'), left_on='zfish_id', right_index=True, how='inner')
    if agg == 'maxabs':
        merged['absstat'] = merged['stat'].abs()
        merged = merged.sort_values('absstat', ascending=False).drop_duplicates('human_symbol', keep='first')
        collapsed = merged.set_index('human_symbol')['stat']
    else:
        collapsed = merged.groupby('human_symbol')['stat'].mean()
    return collapsed.astype(float)

def get_collectri_net():
    """Load the full CollecTRI regulons (human), across decoupler v1.x and v2.x."""
    import decoupler as dc
    if hasattr(dc, 'get_collectri'): # decoupler 1.x
        network = dc.get_collectri(organism='human', split_complexes=False)
    elif hasattr(dc, 'op') and hasattr(dc.op, 'collectri'): # decoupler 2.x
        network = dc.op.collectri(organism='human')
    else:
        raise RuntimeError("This decoupler version exposes no CollecTRI loader (expected dc.get_collectri or dc.op.collectri).")
    if 'weight' not in network.columns and 'mor' in network.columns:
        network = network.rename(columns={'mor': 'weight'})
    return network

def run_tf_activity(human_stat, collectri, label):
    """Infer activity for every regulon TF from one gene-level statistic vector."""
    import decoupler as dc
    sample_by_gene = human_stat.rename(label).to_frame().T.astype(float)
    if hasattr(dc, 'run_ulm'): # decoupler 1.x
        activities, pvalues = dc.run_ulm(mat=sample_by_gene, net=collectri, source='source', target='target', weight='weight', min_n=5, verbose=True)
    elif hasattr(dc, 'mt') and hasattr(dc.mt, 'ulm'): # decoupler 2.x
        activities, pvalues = dc.mt.ulm(data=sample_by_gene, net=collectri)
    else:
        raise RuntimeError("This decoupler version exposes no ulm method (expected dc.run_ulm or dc.mt.ulm).")
    result = pd.DataFrame({'activity': activities.loc[label], 'pval': pvalues.loc[label]})
    return result.sort_values('activity', ascending=False)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Full TF-activity comparison: mutant (Q1) versus natural ageing (Q2)")
    print()

    ortho = load_orthology_lookup(ORTHOLOGY_MAP_CSV)

    # Full CollecTRI regulons (downloaded once; reused for Q2 and, if asked, Q1)
    print("Downloading CollecTRI regulons (human) via decoupler/OmniPath...")
    collectri = get_collectri_net()
    print("  CollecTRI: %d TFs, %d interactions." % (collectri['source'].nunique(), len(collectri)))
    print()

    # Q2: score natural ageing from scratch across all regulon TFs
    print("Scoring natural ageing (Q2) across all CollecTRI TFs...")
    ageing_vector = build_human_stat_vector(AGE_DEG_CSV, ortho, STAT_COL, agg=AGG)
    print("  Human gene-level vector (ageing): %d symbols." % len(ageing_vector))
    ageing_activity = run_tf_activity(ageing_vector, collectri, label='age_M37_vs_M11')
    ageing_activity.to_csv(os.path.join(OUTPUT_DIR, 'tf_activity_age.csv'))
    n_sig_age = int((ageing_activity['pval'] < SIG_TF_PVAL).sum())
    print("  Saved: tf_activity_age.csv")
    print("  TFs scored: %d; significant (p<%.2f): %d" % (len(ageing_activity), SIG_TF_PVAL, n_sig_age))
    print("  Most activated in ageing: %s" % ", ".join(ageing_activity.head(8).index))
    print("  Most repressed in ageing: %s" % ", ".join(ageing_activity.tail(8).index[::-1]))
    print()

    # Q1: full mutant TF activities, loaded from the existing table
    mutant_activity = pd.read_csv(TF_ACTIVITY_Q1_CSV, index_col=0)
    n_sig_mut = int((mutant_activity['pval'] < SIG_TF_PVAL).sum())
    print("Q1 mutant TF activities: %d TFs, %d significant." % (len(mutant_activity), n_sig_mut))
    print()

    # Full versus full comparison (every TF scored independently in each contrast)
    comparison = pd.DataFrame({'q1_activity': mutant_activity['activity'], 'q1_pval': mutant_activity['pval'], 'age_activity': ageing_activity['activity'], 'age_pval': ageing_activity['pval'],}).dropna(subset=['q1_activity', 'age_activity'])
    comparison['sig_q1'] = comparison['q1_pval'] < SIG_TF_PVAL
    comparison['sig_age'] = comparison['age_pval'] < SIG_TF_PVAL
    comparison['sig_both'] = comparison['sig_q1'] & comparison['sig_age']
    comparison['same_direction'] = np.sign(comparison['q1_activity']) == np.sign(comparison['age_activity'])

    r_all, p_all = pearsonr(comparison['q1_activity'], comparison['age_activity'])
    both = comparison[comparison['sig_both']]
    print("Comparison over %d TFs scored in both contrasts:" % len(comparison))
    print("  Pearson r = %.3f (p = %.1e)" % (r_all, p_all))
    print("  Same direction: %d of %d TFs (%.0f%%)" % (comparison['same_direction'].sum(), len(comparison), 100 * comparison['same_direction'].mean()))
    print("  Significant in BOTH: %d TFs" % len(both))
    if len(both):
        print("    of those, same direction: %.0f%%" % (100 * both['same_direction'].mean()))
    print()

    shown = [tf for tf in KEY_TFS if tf in comparison.index]
    if shown:
        print("Key transcription factors (positive = activated, negative = repressed):")
        for tf in shown:
            row = comparison.loc[tf]
            tag = 'same direction' if row['same_direction'] else 'different direction'
            print("  %-6s mutant %+.3f (p=%.3f)   ageing %+.3f (p=%.3f)   (%s)" % (tf, row['q1_activity'], row['q1_pval'], row['age_activity'], row['age_pval'], tag))
        print()

    comparison.sort_values('q1_activity').to_csv(os.path.join(OUTPUT_DIR, 'q1_vs_age_tf_comparison.csv'))
    print("Saved: q1_vs_age_tf_comparison.csv")

    # Figure: Q1 versus ageing TF activity, TFs significant in both are labelled
    fig, ax = plt.subplots(figsize=(7, 6))
    colours = ['#E63946' if s else '#BBBBBB' for s in comparison['sig_both']]
    ax.scatter(comparison['q1_activity'], comparison['age_activity'], c=colours, s=26, alpha=0.7, edgecolors='none')
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    label_set = both.reindex(both['q1_activity'].abs().sort_values(ascending=False).index).head(20)
    for tf, row in label_set.iterrows():
        ax.annotate(str(tf), (row['q1_activity'], row['age_activity']), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('TF activity in mutant (Q1 genotype)')
    ax.set_ylabel('TF activity in natural ageing (Q2 age)')
    ax.set_title('Full TF-activity comparison: mutant versus ageing\n(r = %.2f; red = significant in both)' % r_all, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q1_vs_age_scatter.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: q1_vs_age_scatter.png")
    print()
    print("Outputs in: %s" % OUTPUT_DIR)

if __name__ == '__main__':
    main()