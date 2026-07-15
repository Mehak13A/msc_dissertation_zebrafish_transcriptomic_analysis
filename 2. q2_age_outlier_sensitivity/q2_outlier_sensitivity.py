#!/usr/bin/env python3
"""
2. Q2 Age-Effect Outlier Sensitivity Analysis (bulk RNA-seq, zebrafish muscle)

PURPOSE
    Test whether the Q2 (age) set of differentially expressed genes (DEGs) is driven by the three technical-outlier
    samples flagged on the PCA: M37_WT_1, M37_WT_3 (old) and M11_WT_4 (young). In other words: is the ageing signal real,
    or an artefact of a few unusual samples?

STRATEGY
    Re-run the Q2 DESeq2 contrast (old M37_WT vs young M11_WT) twice, on the SAME filtered gene universe:
        FULL   : all 11 wild-type samples (6 old + 5 young)  -> reproduces the q2_age_DEG_results.csv produced by the main DE pipeline.
        REDUCED: 8 wild-type samples (4 old + 4 young), after dropping the trio.

    Dropping samples also lowers statistical power, so a fall in the raw DEG count is expected even if the outliers
    are harmless. The decisive robustness evidence is therefore the CONCORDANCE of log2 fold changes between the two
    runs: if effect sizes stay tightly correlated and the top hits persist, the trio is not steering the biology.

    Conventions are kept identical to differential_expression_analysis.py:
      - pre-filter: >= 10 reads in >= 3 samples of the full 20-sample matrix
      - Storey q-values (same estimator); significance at q < 0.05 and |log2FC| > 1
      - DESeq2 via pydeseq2, contrast = [age_group, old, young]

INPUT DATASET
    gene_count.xls
        Tab-delimited raw read-count matrix with a gene_id column, 20 sample columns (M18_MUT_1 ... M11_WT_5) and
        gene-annotation columns. Only the wild-type age samples (M37_WT_*, M11_WT_*) are used here.

OUTPUTS (written to the output directory)
    q2_outlier_sensitivity_comparison.csv
        Per-gene comparison for genes significant in EITHER run: log2 fold change and q-value from each run,
        significance flags, and whether the direction agrees.
    q2_age_DEG_results_no_outliers.csv
        Full DESeq2 results table for the REDUCED (outliers-dropped) run.
    q2_outlier_sensitivity.png
        Two-panel figure: (left) full-vs-reduced log2FC concordance scatter; (right) DEG counts (up/down) for the
        full vs reduced runs.
    Console
        Overlap statistics, log2FC concordance (Pearson/Spearman), and a top-20 stability check.

REFERENCES (this file only)
    These references apply to this source file only and are independent of any reference numbering used in the accompanying report.

    [1] M. I. Love, W. Huber, and S. Anders, "Moderated estimation of fold change and dispersion for RNA-seq data
        with DESeq2", Genome Biology, vol. 15, no. 12, art. 550, 2014.
    [2] B. Muzellec, M. Telenczuk, V. Cabeli, and M. Andreux, "PyDESeq2: a python package for bulk RNA-seq differential
        expression analysis", Bioinformatics, vol. 39, no. 9, 2023.
    [3] J. D. Storey and R. Tibshirani, "Statistical significance for genomewide studies", Proc. Nat. Acad. Sci.,
        vol. 100, no. 16, pp. 9440-9445, 2003.
"""

# Import all necessary libraries and modules

import os
import argparse
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') # non-interactive backend: render figures to files
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pydeseq2.dds import DeseqDataSet # DESeq2 dataset object (fits the model)
from pydeseq2.ds import DeseqStats # DESeq2 statistics object (Wald test, results)

warnings.filterwarnings('ignore') # suppress non-critical library warnings for a clean log

# Configuration (kept consistent with differential_expression_analysis.py)

# Input/output locations as per file paths on my computer. These can be overridden on the command line.
# Input: tab-delimited raw-count matrix
DEFAULT_COUNT_MATRIX_PATH = '/Users/mehakagrawal/Desktop/Final_Dissertation/Datasets/adult_muscle_18mo/gene_count.xls'
# Output: where CSVs and the figure are written
DEFAULT_OUTPUT_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/2. q2_age_outlier_sensitivity'

# Significance thresholds: q-value below 0.05 (5% FDR) AND |log2 fold change| > 1.
QVALUE_CUTOFF = 0.05
LOG2FC_CUTOFF = 1.0

# The three technical-outlier samples flagged on the PCA (the trio being tested).
OUTLIER_SAMPLES = ['M37_WT_1', 'M37_WT_3', 'M11_WT_4']

# Pre-filter thresholds: keep a gene with >= MIN_READS_PER_GENE reads in >= MIN_SAMPLES_EXPRESSING samples (applied to the full 20-sample matrix).
MIN_READS_PER_GENE = 10
MIN_SAMPLES_EXPRESSING = 3

# Storey q-value helper
def storey_qvalues(pvals):
    """Compute Storey & Tibshirani q-values [3] from a vector of p-values.

    A q-value is the minimum false discovery rate at which a test can be called significant. It equals pi0 * (Benjamini-Hochberg-adjusted p-value),
    where pi0 is the estimated proportion of truly-null genes. This is the same estimator used in the main DE pipeline, so the two are directly comparable.

    ATTRIBUTION: implemented from scratch following the q-value procedure of Storey & Tibshirani [3]. The smoothing-spline estimate
    of pi0 follows that paper. Returns (q-values aligned to the input order, estimated pi0).
    """
    pvals_arr = np.asarray(pvals, dtype=float)
    valid_mask = ~np.isnan(pvals_arr) # ignore genes with no p-value
    valid_pvals = pvals_arr[valid_mask]
    n_valid = len(valid_pvals)
    if n_valid == 0:
        return np.full_like(pvals_arr, np.nan), np.nan

    # Estimate pi0 (fraction of true nulls) across a grid of lambda thresholds.
    lambdas = np.arange(0.05, 0.96, 0.05)
    pi0s = [np.mean(valid_pvals > lam) / (1.0 - lam) for lam in lambdas]
    try:
        # Smoothing-spline estimate of pi0 at the largest lambda, as in [3].
        # Imported here so the function still runs (falling back below) without SciPy.
        from scipy.interpolate import UnivariateSpline
        pi0 = float(UnivariateSpline(lambdas, pi0s, k=3, s=0)(lambdas[-1]))
    except Exception:
        pi0 = pi0s[-1]
    pi0 = min(max(pi0, 1e-8), 1.0) # clamp to a sensible (0, 1] range

    # Convert sorted p-values to q-values, then enforce monotonicity.
    ascending_order = np.argsort(valid_pvals)
    pvals_sorted = valid_pvals[ascending_order]
    qvals_sorted = pi0 * n_valid * pvals_sorted / np.arange(1, n_valid + 1)
    qvals_sorted = np.minimum.accumulate(qvals_sorted[::-1])[::-1]
    qvals_sorted = np.minimum(qvals_sorted, 1.0)

    # Scatter q-values back to the original gene order.
    qvals_ranked = np.empty(n_valid)
    qvals_ranked[ascending_order] = qvals_sorted
    qvals_full = np.full_like(pvals_arr, np.nan)
    qvals_full[valid_mask] = qvals_ranked
    return qvals_full, pi0

# Run the Q2 (age) DESeq2 contrast on a chosen subset of samples

def run_age_contrast(filtered_counts, gene_name_map, sample_subset, run_label):
    """Run the Q2 old-vs-young DESeq2 contrast [1], [2] on a given sample subset.

    Parameters
    filtered_counts : DataFrame
        Pre-filtered count matrix (genes x samples).
    gene_name_map : dict
        Mapping gene_id -> readable gene name.
    sample_subset : list of str
        Sample columns to include (a mix of old M37_WT and young M11_WT).
    run_label : str
        Human-readable label for the console output (e.g. 'FULL', 'REDUCED').

    Returns
    tuple(DataFrame, DataFrame)
        The full results table and the subset of significant DEGs.
    """
    print("\n")
    print(f"DESeq2 - Q2 ({run_label}): {len(sample_subset)} samples")
    n_old = sum(s.startswith('M37') for s in sample_subset)
    n_young = sum(s.startswith('M11') for s in sample_subset)
    print(f"  old (M37_WT)={n_old}   young (M11_WT)={n_young}")

    # Build the samples x genes count matrix and the age-group metadata.
    subset_counts = filtered_counts[sample_subset].T.astype(int)
    subset_metadata = pd.DataFrame(index=sample_subset)
    subset_metadata['age_group'] = ['old' if 'M37' in s else 'young' for s in sample_subset]

    # Fit the DESeq2 model and run the old-vs-young Wald test [1], [2].
    dds = DeseqDataSet(counts=subset_counts, metadata=subset_metadata, design="~age_group")
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["age_group", "old", "young"])
    stats.summary()

    # Collect results and add readable names + Storey q-values.
    results = stats.results_df.copy()
    results['gene_name'] = results.index.map(gene_name_map)
    results['qvalue'], pi0 = storey_qvalues(results['pvalue'])

    # Apply the significance thresholds.
    significant = results[(results['qvalue'] < QVALUE_CUTOFF) & (results['log2FoldChange'].abs() > LOG2FC_CUTOFF)]
    n_up = (significant['log2FoldChange'] > 0).sum()
    n_down = (significant['log2FoldChange'] < 0).sum()
    print(f"  DEGs (q<{QVALUE_CUTOFF}, |log2FC|>{LOG2FC_CUTOFF}): {len(significant)}  " f"(up in old={n_up}, down in old={n_down})  [pi0={pi0:.3f}]")
    return results, significant

# Main workflow

def main():
    # Command-line arguments (fall back to the default paths above).
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=DEFAULT_COUNT_MATRIX_PATH)
    parser.add_argument('--outdir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load and filter exactly as the main pipeline does
    raw_table = pd.read_csv(args.input, sep='\t')
    sample_columns = [c for c in raw_table.columns if c.startswith('M')]
    gene_name_map = raw_table.set_index('gene_id')['gene_name'].to_dict()

    count_matrix = raw_table[['gene_id'] + sample_columns].copy().set_index('gene_id')
    genes_pass_filter = (count_matrix >= MIN_READS_PER_GENE).sum(axis=1) >= MIN_SAMPLES_EXPRESSING
    filtered_counts = count_matrix.loc[genes_pass_filter]
    print(f"Gene universe after filtering: {filtered_counts.shape[0]} genes " f"(shared by both runs)")

    def gene_label(gene_id):
        """Readable gene name for a gene_id, or a truncated Ensembl ID if unnamed."""
        gene_name_val = gene_name_map.get(gene_id)
        return gene_name_val if (isinstance(gene_name_val, str) and gene_name_val.strip()) else gene_id[:18]

    # Define the two sample sets: full (with trio) and reduced (trio dropped)
    age_samples_full = [c for c in sample_columns if c.startswith('M37') or c.startswith('M11')]
    missing_outliers = [o for o in OUTLIER_SAMPLES if o not in age_samples_full]
    if missing_outliers:
        raise ValueError(f"Outlier sample(s) not found in data: {missing_outliers}")
    age_samples_reduced = [c for c in age_samples_full if c not in OUTLIER_SAMPLES]

    # Run both contrasts on the same gene universe
    results_full, significant_full = run_age_contrast(filtered_counts, gene_name_map, age_samples_full, 'FULL / with outliers')
    results_reduced, significant_reduced = run_age_contrast(filtered_counts, gene_name_map, age_samples_reduced, 'REDUCED / outliers dropped')

    deg_set_full, deg_set_reduced = set(significant_full.index), set(significant_reduced.index)
    shared_degs = deg_set_full & deg_set_reduced
    union_degs = deg_set_full | deg_set_reduced

    # Overlap / recovery statistics
    print("\n")
    print("DEG SET COMPARISON")
    print(f"  Full DEGs        : {len(deg_set_full)}")
    print(f"  Reduced DEGs     : {len(deg_set_reduced)}")
    print(f"  Shared           : {len(shared_degs)}")
    print(f"  Full-only (lost) : {len(deg_set_full - deg_set_reduced)}")
    print(f"  Reduced-only(new): {len(deg_set_reduced - deg_set_full)}")
    print(f"  Jaccard          : {len(shared_degs)/len(union_degs):.3f}")
    if len(deg_set_full):
        print(f"  Recovery of full set in reduced run: {100*len(shared_degs)/len(deg_set_full):.1f}%")

    # log2 fold-change concordance (the key robustness metric)
    # Compare effect sizes for every gene significant in EITHER run.
    union_gene_ids = sorted(union_degs)
    comparison_table = pd.DataFrame({
        'gene_name':      [gene_label(g) for g in union_gene_ids],
        'log2FC_full':    results_full.loc[union_gene_ids, 'log2FoldChange'].values,
        'log2FC_reduced': results_reduced.loc[union_gene_ids, 'log2FoldChange'].values,
        'qvalue_full':    results_full.loc[union_gene_ids, 'qvalue'].values,
        'qvalue_reduced': results_reduced.loc[union_gene_ids, 'qvalue'].values, }, index=union_gene_ids)
    comparison_table['sig_full'] = comparison_table.index.isin(deg_set_full)
    comparison_table['sig_reduced'] = comparison_table.index.isin(deg_set_reduced)
    comparison_table['same_direction'] = (comparison_table['log2FC_full'] > 0) == (comparison_table['log2FC_reduced'] > 0)

    comparison_valid = comparison_table.dropna(subset=['log2FC_full', 'log2FC_reduced'])
    pearson_r = comparison_valid['log2FC_full'].corr(comparison_valid['log2FC_reduced'], method='pearson')
    spearman_r = comparison_valid['log2FC_full'].corr(comparison_valid['log2FC_reduced'], method='spearman')
    same_direction_pct = 100 * comparison_valid['same_direction'].mean()
    print("\n")
    print("log2FC CONCORDANCE (genes significant in EITHER run)")
    print(f"  Pearson r  : {pearson_r:.3f}")
    print(f"  Spearman r : {spearman_r:.3f}")
    print(f"  Same sign  : {same_direction_pct:.1f}%")

    # Save the comparison table and the reduced-run results.
    comparison_table.sort_values('qvalue_full').to_csv(os.path.join(args.outdir, 'q2_outlier_sensitivity_comparison.csv'))
    results_reduced.to_csv(os.path.join(args.outdir, 'q2_age_DEG_results_no_outliers.csv'))

    # Top-hit stability: are the top-20 full-run DEGs still significant?
    top20_full_hits = significant_full.sort_values('qvalue').head(20).index
    print("\n")
    print("TOP-20 FULL-RUN HITS: still significant after dropping outliers?")
    n_kept = 0
    for gene_id in top20_full_hits:
        still_significant = gene_id in deg_set_reduced
        n_kept += still_significant
        fc_full = results_full.loc[gene_id, 'log2FoldChange']
        fc_reduced = results_reduced.loc[gene_id, 'log2FoldChange']
        status_flag = 'kept ' if still_significant else 'LOST '
        print(f"  [{status_flag}] {gene_label(gene_id):20s} log2FC {fc_full:+.2f} -> {fc_reduced:+.2f}  " f"q_red={results_reduced.loc[gene_id,'qvalue']:.2e}")
    print(f"\n  {n_kept}/20 of the top hits remain significant in the reduced run.")

    # Figure: concordance scatter (left) + DEG-count bars (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left panel: colour each gene by which run(s) it is significant in.
    point_colours = np.where(comparison_valid['sig_full'] & comparison_valid['sig_reduced'], '#2A9D8F',
                     np.where(comparison_valid['sig_full'] & ~comparison_valid['sig_reduced'], '#E63946',
                     np.where(~comparison_valid['sig_full'] & comparison_valid['sig_reduced'], '#457B9D', '#CCCCCC')))
    ax1.scatter(comparison_valid['log2FC_full'], comparison_valid['log2FC_reduced'], c=point_colours, s=14, alpha=0.7, edgecolors='none')
    axis_limit = np.nanmax(np.abs([comparison_valid['log2FC_full'], comparison_valid['log2FC_reduced']])) * 1.05
    ax1.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], color='grey', ls='--', lw=0.8)  # y=x reference
    ax1.axhline(0, color='grey', lw=0.5)
    ax1.axvline(0, color='grey', lw=0.5)
    ax1.set_xlim(-axis_limit, axis_limit)
    ax1.set_ylim(-axis_limit, axis_limit)
    ax1.set_xlabel('log2FC - full (11 samples)')
    ax1.set_ylabel('log2FC - reduced (8 samples)')
    ax1.set_title(f'Effect-size concordance\nPearson r={pearson_r:.3f}, same sign={same_direction_pct:.1f}%', fontweight='bold')
    legend_handles = [Line2D([0], [0], marker='o', ls='', color='#2A9D8F', label='sig in both'),
                      Line2D([0], [0], marker='o', ls='', color='#E63946', label='sig full only'),
                      Line2D([0], [0], marker='o', ls='', color='#457B9D', label='sig reduced only')]
    ax1.legend(handles=legend_handles, fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.2)

    # Right panel: stacked bar of up/down DEG counts for each run.
    bar_labels = ['Full\n(11)', 'Reduced\n(8)']
    up_counts = [(significant_full['log2FoldChange'] > 0).sum(), (significant_reduced['log2FoldChange'] > 0).sum()]
    down_counts = [(significant_full['log2FoldChange'] < 0).sum(), (significant_reduced['log2FoldChange'] < 0).sum()]
    bar_positions = np.arange(2)
    ax2.bar(bar_positions, up_counts, 0.55, label='up in old', color='#E63946', alpha=0.85)
    ax2.bar(bar_positions, down_counts, 0.55, bottom=up_counts, label='down in old', color='#457B9D', alpha=0.85)
    for i in range(2):
        ax2.text(i, up_counts[i] + down_counts[i] + max(up_counts + down_counts) * 0.01, f'{up_counts[i]+down_counts[i]}', ha='center', fontweight='bold')
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(bar_labels)
    ax2.set_ylabel('Significant DEGs')
    ax2.set_title('Q2 DEG count: with vs without outliers', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    figure_path = os.path.join(args.outdir, 'q2_outlier_sensitivity.png')
    plt.savefig(figure_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure: {figure_path}")
    print("Saved: q2_outlier_sensitivity_comparison.csv")
    print("Saved: q2_age_DEG_results_no_outliers.csv")
    print("\n Sensitivity analysis complete.")

if __name__ == '__main__':
    main()