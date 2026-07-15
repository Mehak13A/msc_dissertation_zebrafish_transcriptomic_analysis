#!/usr/bin/env python3
"""
4. tert 3-month DE Analysis Pipeline (bulk RNA-seq, zebrafish muscle)

PURPOSE
    Differential expression (DE) analysis of the 3-month-old zebrafish muscle dataset (JenAge tert_3month), where
    telomerase mutants are NOT yet expected to show a phenotype. It provides the "young" timepoint that complements the
    18-month adult analysis. Three genotype contrasts are run:
        A: MUT vs WT   (tert -/- vs tert +/+)
        B: Het vs WT   (tert +/- vs tert +/+)
        C: MUT vs Het  (tert -/- vs tert +/-)
    Additionally, the 3-month MUT-vs-WT result is compared against the 18-month genotype result to check how the effect
    changes with age.

INPUT DATASETS
    13 featureCounts files (one per sample) named Muscle_<animalID>_fastq-featureCounts_gene.txt in COUNTS_DIR.
    Each is a tab-delimited featureCounts table with a 'Geneid' column, annotation columns, and a final read-count column.
    The 13 animals are: 5 wild-type (WT), 5 heterozygous (Het) and 3 mutant (MUT).

    (Additional) q1_genotype_DEG_results.csv
        The 18-month genotype DE table (output of the main DE pipeline). Used only for the cross-dataset comparison in STEP 7.

OUTPUTS (all written to OUTPUT_DIR)
    tert3m_raw_counts.csv         : combined raw count matrix (genes x 13 samples)
    tert3m_pca.png                : PCA (PC1-PC2 and PC1-PC3) coloured by genotype
    tert3m_MUT_vs_WT_results.csv  : DE results, contrast A
    tert3m_Het_vs_WT_results.csv  : DE results, contrast B
    tert3m_MUT_vs_Het_results.csv : DE results, contrast C
    volcano_MUT_vs_WT.png, volcano_Het_vs_WT.png, volcano_MUT_vs_Het.png
    cross_dataset_overlap_3m_vs_18m.csv : shared DEGs between 3mo and 18mo (18mo CSV provided given an overlap exists)
    Console                       : per-step progress and a final summary.

REFERENCES (this file only)
    These references apply to this source file only and are independent of any reference numbering used in the accompanying report.

    [1] M. I. Love, W. Huber, and S. Anders, "Moderated estimation of fold change and dispersion for RNA-seq data with
        DESeq2", Genome Biology, vol. 15, no. 12, art. 550, 2014.
    [2] B. Muzellec, M. Telenczuk, V. Cabeli, and M. Andreux, "PyDESeq2: a python package for bulk RNA-seq differential
        expression analysis", Bioinformatics, vol. 39, no. 9, 2023.
    [3] J. D. Storey and R. Tibshirani, "Statistical significance for genomewide studies", Proc. Nat. Acad. Sci.,
        vol. 100, no. 16, pp. 9440-9445, 2003.
    [4] F. Pedregosa et al., "Scikit-learn: machine learning in Python", J. Mach. Learn. Res., vol. 12, pp. 2825-2830,
        2011.
"""

# Import all necessary libraries and modules

import os
import glob
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') # non-interactive backend: render figures to files
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA # PCA for sample structure [4]
from pydeseq2.dds import DeseqDataSet # DESeq2 dataset object [1], [2]
from pydeseq2.ds import DeseqStats # DESeq2 statistics (Wald test) [1], [2]
from pydeseq2.preprocessing import deseq2_norm # median-of-ratios normalisation [1], [2]
from adjustText import adjust_text # non-overlapping plot labels [5]

warnings.filterwarnings('ignore') # suppress non-critical library warnings

# Configuration (Input and output paths as per my computer)

# Folder containing the 13 Muscle_*_fastq-featureCounts_gene.txt files.
COUNTS_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Datasets/tert_3mo/13_files'

# Output directory
OUTPUT_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/4. tert_3month_de_analysis'

# Path to the 18-month genotype DE results (from the main pipeline), for the cross-dataset comparison in STEP 7.
Q1_18MONTH_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/1. differential_expression_analysis/q1_genotype_DEG_results.csv'

# Significance thresholds are kept identical to differential_expression_analysis.py so the two datasets are analysed on the same footing (Storey q-values, not padj).
QVALUE_CUTOFF = 0.05
LOG2FC_CUTOFF = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Storey q-value helper

def storey_qvalues(pvals):
    """Compute Storey & Tibshirani q-values [3] from a vector of p-values.

    A q-value is the minimum false discovery rate at which a test can be called significant. It equals pi0 * (Benjamini-Hochberg-adjusted p-value),
    where pi0 is the estimated proportion of truly-null genes. Same estimator used across the project, so results are directly comparable.

    ATTRIBUTION: implemented from scratch following the q-value procedure of Storey & Tibshirani [3].
    Returns (q-values aligned to input, estimated pi0).
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
        # Smoothing-spline estimate of pi0 at the largest lambda, as in [3]. Imported here so the function still runs (falling back below) without SciPy.
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

# STEP 1 - Build the combined count matrix from the 13 featureCounts files

# Each sample was quantified separately (one featureCounts file per animal). Read the count column from each file and
# join them into a single genes x samples matrix. The animal ID in the file name determines the genotype.
print("STEP 1: Building combined count matrix from 13 featureCounts files")

# Which genotype each animal ID belongs to (5 WT, 5 Het, 3 MUT).
genotype_by_animal_id = {'234': 'WT', '253': 'WT', '255': 'WT', '257': 'WT', '270': 'WT', '235': 'Het', '251': 'Het', '258': 'Het',
                         '264': 'Het', '271': 'Het', '254': 'MUT', '256': 'MUT', '273': 'MUT',
}

featurecounts_files = sorted(glob.glob(os.path.join(COUNTS_DIR, '*.txt')))
print(f"Found {len(featurecounts_files)} featureCounts files")

if len(featurecounts_files) == 0:
    print(f"\nERROR: No featureCounts files found in:\n  {COUNTS_DIR}")
    print("Check that COUNTS_DIR points to the folder containing the 13 Muscle_*.txt files.")
    exit(1)

counts_by_sample = {}
for file_path in featurecounts_files:
    file_name = os.path.basename(file_path)
    animal_id = file_name.split('_')[1] # e.g. 'Muscle_234_...' -> '234'
    featurecounts_table = pd.read_csv(file_path, sep='\t', comment='#')
    count_column = featurecounts_table.columns[-1] # featureCounts puts counts in the last column
    sample_name = f"M3_{genotype_by_animal_id[animal_id]}_{animal_id}" # e.g. 'M3_WT_234'
    counts_by_sample[sample_name] = featurecounts_table.set_index('Geneid')[count_column]
    print(f"  {sample_name}: {featurecounts_table[count_column].sum()/1e6:.1f}M reads mapped")

count_matrix = pd.DataFrame(counts_by_sample)
print(f"\nCombined count matrix: {count_matrix.shape[0]} genes x {count_matrix.shape[1]} samples")
count_matrix.to_csv(os.path.join(OUTPUT_DIR, 'tert3m_raw_counts.csv'))
print("Saved: tert3m_raw_counts.csv")

# STEP 2 - Build sample metadata

# Derive genotype and animal ID from each sample name (all animals are 3 months).
print("\n")
print("STEP 2: Metadata")

sample_columns = count_matrix.columns.tolist()
sample_metadata = pd.DataFrame(index=sample_columns)
sample_metadata['animal_id'] = [c.split('_')[-1] for c in sample_columns]
sample_metadata['genotype'] = [c.split('_')[1] for c in sample_columns]
sample_metadata['age'] = '3mo'
print(sample_metadata)
print(f"\nGroups: {sample_metadata['genotype'].value_counts().to_dict()}")

# STEP 3 - Pre-filter low-count genes

# Keep a gene only if it has >= min_reads_per_gene reads in >= min_samples_expressing samples
# (same rule as the main pipeline). Removes near-silent, noisy genes.
print("\n")
print("STEP 3: Pre-filtering")

min_reads_per_gene, min_samples_expressing = 10, 3
genes_pass_filter = (count_matrix >= min_reads_per_gene).sum(axis=1) >= min_samples_expressing
filtered_counts = count_matrix.loc[genes_pass_filter]
print(f"Before: {count_matrix.shape[0]} genes")
print(f"After (>= {min_reads_per_gene} reads in >= {min_samples_expressing} samples): {filtered_counts.shape[0]} genes")
print(f"Removed: {count_matrix.shape[0] - filtered_counts.shape[0]}")

# STEP 4 - Normalisation and PCA

# Normalise (DESeq2 median-of-ratios), log2-transform, and run PCA to see how the genotypes separate. At 3 months the mutants are not expected to separate from WT.
print("\n")
print("STEP 4: Normalisation + PCA")

filtered_counts_T = filtered_counts.T.astype(int) # samples x genes
normalised_counts, _ = deseq2_norm(filtered_counts_T) # median-of-ratios [1], [2]
log_norm_expr = np.log2(normalised_counts + 1)

pca_model = PCA(n_components=3) # PCA [4]
pca_coords = pca_model.fit_transform(log_norm_expr)
pca_scores = pd.DataFrame(pca_coords, columns=['PC1', 'PC2', 'PC3'], index=sample_columns)
pca_scores['genotype'] = sample_metadata['genotype']

# Fixed colour and marker per genotype.
genotype_colours = {'WT': '#457B9D', 'Het': '#E9C46A', 'MUT': '#E63946'}
genotype_markers = {'WT': 'o', 'Het': 's', 'MUT': 'D'}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for genotype in ['WT', 'Het', 'MUT']:
    group_mask = pca_scores['genotype'] == genotype
    axes[0].scatter(pca_scores.loc[group_mask, 'PC1'], pca_scores.loc[group_mask, 'PC2'], c=genotype_colours[genotype], marker=genotype_markers[genotype], s=90,
                    label=genotype, edgecolors='k', linewidth=0.5, zorder=3)
    axes[1].scatter(pca_scores.loc[group_mask, 'PC1'], pca_scores.loc[group_mask, 'PC3'], c=genotype_colours[genotype], marker=genotype_markers[genotype], s=90,
                    label=genotype, edgecolors='k', linewidth=0.5, zorder=3)

axes[0].set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title('tert_3month PCA: PC1 vs PC2'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC3 ({pca_model.explained_variance_ratio_[2]*100:.1f}%)')
axes[1].set_title('tert_3month PCA: PC1 vs PC3'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tert3m_pca.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"PCA variance: PC1={pca_model.explained_variance_ratio_[0]*100:.1f}%, " f"PC2={pca_model.explained_variance_ratio_[1]*100:.1f}%, " f"PC3={pca_model.explained_variance_ratio_[2]*100:.1f}%")

# STEP 5 - DE analysis: three genotype contrasts

def run_de_contrast(filtered_counts, sample_metadata, sample_subset, design_formula, contrast, label):
    """Run one DESeq2 contrast [1], [2] on a subset of samples.

    Fits the negative-binomial model, runs the Wald test for `contrast`, adds Storey q-values, applies the significance
    thresholds, and prints a summary. Returns (full results table, significant DEGs).
    """
    print(f"\n")
    print(f"DE Analysis: {label}")

    subset_counts = filtered_counts[sample_subset].T.astype(int) # samples x genes
    subset_metadata = sample_metadata.loc[sample_subset].copy()

    dds = DeseqDataSet(counts=subset_counts, metadata=subset_metadata, design=design_formula)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=contrast)
    stats.summary()

    results = stats.results_df.copy()
    results['qvalue'], pi0 = storey_qvalues(results['pvalue'])
    significant = results[(results['qvalue'] < QVALUE_CUTOFF) & (results['log2FoldChange'].abs() > LOG2FC_CUTOFF)]
    print(f"\nSignificant DEGs (q<{QVALUE_CUTOFF}, |log2FC|>{LOG2FC_CUTOFF}): {len(significant)}   [pi0={pi0:.3f}]")
    print(f"  Up: {(significant['log2FoldChange']>0).sum()}")
    print(f"  Down: {(significant['log2FoldChange']<0).sum()}")
    print(f"\nTop 15 by q-value:")
    for _, row in significant.sort_values('qvalue').head(15).iterrows():
        print(f"  {row.name[:25]:<25s}  log2FC={row['log2FoldChange']:+.2f}  qvalue={row['qvalue']:.2e}")
    return results, significant

# Contrast A: mutant vs wild-type.
mut_vs_wt_samples = [s for s in sample_columns if 'MUT' in s or 'WT' in s]
mut_wt_results, mut_wt_sig = run_de_contrast(filtered_counts, sample_metadata, mut_vs_wt_samples, "~genotype", ["genotype", "MUT", "WT"], "A: MUT vs WT (3mo)")
mut_wt_results.to_csv(os.path.join(OUTPUT_DIR, 'tert3m_MUT_vs_WT_results.csv'))

# Contrast B: heterozygous vs wild-type.
het_vs_wt_samples = [s for s in sample_columns if 'Het' in s or 'WT' in s]
het_wt_results, het_wt_sig = run_de_contrast(filtered_counts, sample_metadata, het_vs_wt_samples, "~genotype", ["genotype", "Het", "WT"], "B: Het vs WT (3mo)")
het_wt_results.to_csv(os.path.join(OUTPUT_DIR, 'tert3m_Het_vs_WT_results.csv'))

# Contrast C: mutant vs heterozygous.
mut_vs_het_samples = [s for s in sample_columns if 'MUT' in s or 'Het' in s]
mut_het_results, mut_het_sig = run_de_contrast(filtered_counts, sample_metadata, mut_vs_het_samples, "~genotype", ["genotype", "MUT", "Het"], "C: MUT vs Het (3mo)")
mut_het_results.to_csv(os.path.join(OUTPUT_DIR, 'tert3m_MUT_vs_Het_results.csv'))

# STEP 6 - Volcano plots

# One volcano plot per contrast: fold change (x) vs significance (y).
print("\n")
print("STEP 6: Volcano plots")

def generate_volcano_plot(de_results, title, filename, top_n=12):
    """Draw and save a volcano plot (log2 fold change vs -log10 q-value), colouring significant up/down genes and labelling the top_n."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data = de_results.dropna(subset=['qvalue', 'log2FoldChange']).copy()
    plot_data['-log10q'] = -np.log10(plot_data['qvalue'].clip(lower=1e-300)) # clip avoids log(0)
    significant_up = (plot_data['qvalue'] < QVALUE_CUTOFF) & (plot_data['log2FoldChange'] > LOG2FC_CUTOFF)
    significant_down = (plot_data['qvalue'] < QVALUE_CUTOFF) & (plot_data['log2FoldChange'] < -LOG2FC_CUTOFF)
    not_significant = ~(significant_up | significant_down)

    ax.scatter(plot_data.loc[not_significant, 'log2FoldChange'], plot_data.loc[not_significant, '-log10q'], c='#CCCCCC', s=8, alpha=0.5, label='NS')
    ax.scatter(plot_data.loc[significant_up, 'log2FoldChange'], plot_data.loc[significant_up, '-log10q'], c='#E63946', s=18, alpha=0.7, label=f'Up ({significant_up.sum()})')
    ax.scatter(plot_data.loc[significant_down, 'log2FoldChange'], plot_data.loc[significant_down, '-log10q'], c='#457B9D', s=18, alpha=0.7, label=f'Down ({significant_down.sum()})')

    # Label the most significant genes, de-overlapped with adjustText.
    top_labelled = plot_data[significant_up | significant_down].nlargest(top_n, '-log10q')
    label_texts = [ax.text(row['log2FoldChange'], row['-log10q'], idx[:18], fontsize=7) for idx, row in top_labelled.iterrows()]
    if label_texts:
        adjust_text(label_texts, ax=ax, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))

    ax.axhline(-np.log10(QVALUE_CUTOFF), color='grey', ls='--', lw=0.8)
    ax.axvline(LOG2FC_CUTOFF, color='grey', ls='--', lw=0.8)
    ax.axvline(-LOG2FC_CUTOFF, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('log2 Fold Change'); ax.set_ylabel('-log10 q-value')
    ax.set_title(title, fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

generate_volcano_plot(mut_wt_results, 'tert_3mo: MUT vs WT', 'volcano_MUT_vs_WT.png')
generate_volcano_plot(het_wt_results, 'tert_3mo: Het vs WT', 'volcano_Het_vs_WT.png')
generate_volcano_plot(mut_het_results, 'tert_3mo: MUT vs Het', 'volcano_MUT_vs_Het.png')

# STEP 7 - Cross-dataset gene overlap (3mo vs 18mo genotype effect)

# Compare the 3-month MUT-vs-WT result against the 18-month genotype result to see whether the same genes are affected, and in the same direction, with age.
print("\n")
print("STEP 7: Cross-dataset overlap (3mo vs 18mo genotype effect)")

if Q1_18MONTH_CSV and os.path.exists(Q1_18MONTH_CSV):
    results_18month = pd.read_csv(Q1_18MONTH_CSV, index_col=0)
    # Prefer Storey q-values; fall back to padj only for an older CSV, with a warning.
    if 'qvalue' in results_18month.columns:
        sig_column_18month = 'qvalue'
    else:
        sig_column_18month = 'padj'
        print("WARNING: no 'qvalue' column in the 18mo CSV - falling back to 'padj'.")
        print("         Re-run differential_expression_analysis.py to regenerate it with")
        print("         q-values, otherwise this overlap won't match the canonical DEGs.")

    # Apply the standard significance filter (q < cutoff and |log2FC| > cutoff) to the 18mo results
    significant_18month = results_18month[(results_18month[sig_column_18month] < QVALUE_CUTOFF) & (results_18month['log2FoldChange'].abs() > LOG2FC_CUTOFF)]
    has_gene_names_18month = 'gene_name' in results_18month.columns

    # Compare gene ID coverage between the 3-month and 18-month datasets (they may not test the exact same set of genes due to independent filtering)
    genes_3month = set(mut_wt_results.index)
    genes_18month = set(results_18month.index)
    shared_gene_ids = genes_3month & genes_18month
    print(f"Genes in 3mo dataset: {len(genes_3month)}")
    print(f"Genes in 18mo dataset: {len(genes_18month)}")
    print(f"Shared gene IDs: {len(shared_gene_ids)}")
    print(f"3mo-only genes: {len(genes_3month - genes_18month)}")
    print(f"18mo-only genes: {len(genes_18month - genes_3month)}")

    # Find genes that are significant DEGs in BOTH the 3mo and 18mo MUT vs WT comparisons
    sig_3month_ids = set(mut_wt_sig.index)
    sig_18month_ids = set(significant_18month.index)
    deg_overlap_ids = sig_3month_ids & sig_18month_ids
    print(f"\n3mo MUT vs WT DEGs: {len(sig_3month_ids)}")
    print(f"18mo MUT vs WT DEGs (by {sig_column_18month}): {len(sig_18month_ids)}")
    print(f"DEG overlap: {len(deg_overlap_ids)}")

    if deg_overlap_ids:
        # For overlapping DEGs, check whether the direction of change (up/down) agrees between the two age groups, or goes the opposite way
        n_same_direction = sum(1 for g in deg_overlap_ids if (mut_wt_results.loc[g, 'log2FoldChange'] > 0) == (results_18month.loc[g, 'log2FoldChange'] > 0))
        n_opposite_direction = len(deg_overlap_ids) - n_same_direction
        print(f"  Same direction: {n_same_direction}")
        print(f"  Opposite direction: {n_opposite_direction}")

        overlap_rows = []
        for gene_id in deg_overlap_ids:
            fc_3month = mut_wt_results.loc[gene_id, 'log2FoldChange']
            fc_18month = results_18month.loc[gene_id, 'log2FoldChange']
            overlap_rows.append({
                'gene_id': gene_id,
                'gene_name': results_18month.loc[gene_id, 'gene_name'] if has_gene_names_18month else '',
                'log2FC_3mo': round(fc_3month, 3),
                'qvalue_3mo': mut_wt_results.loc[gene_id, 'qvalue'],
                'log2FC_18mo': round(fc_18month, 3),
                'qvalue_18mo': results_18month.loc[gene_id, sig_column_18month],
                'direction': 'SAME' if (fc_3month > 0) == (fc_18month > 0) else 'OPPOSITE'
            })
        # Sort by 3mo q-value (most significant first) and write out the overlap table
        overlap_df = pd.DataFrame(overlap_rows).sort_values('qvalue_3mo')
        overlap_df.to_csv(os.path.join(OUTPUT_DIR, 'cross_dataset_overlap_3m_vs_18m.csv'), index=False)
        print("  Saved: cross_dataset_overlap_3m_vs_18m.csv")
else:
    if not Q1_18MONTH_CSV:
        print("Q1_18MONTH_CSV not set - skipping cross-dataset comparison.")
    else:
        print(f"18mo Q1 results not found at: {Q1_18MONTH_CSV}")
        print("Skipping cross-dataset comparison.")

# STEP 8 - Summary

print("\n")
print("SUMMARY")
print(f"""
Dataset: 13 muscle samples, 3 months old zebrafish
  WT (tert +/+): n=5
  Het (tert +/-): n=5
  MUT (tert -/-): n=3

Genes analysed: {filtered_counts.shape[0]}

Comparison A - MUT vs WT:
  DEGs: {len(mut_wt_sig)} (up:{(mut_wt_sig['log2FoldChange']>0).sum()} down:{(mut_wt_sig['log2FoldChange']<0).sum()})

Comparison B - Het vs WT:
  DEGs: {len(het_wt_sig)} (up:{(het_wt_sig['log2FoldChange']>0).sum()} down:{(het_wt_sig['log2FoldChange']<0).sum()})

Comparison C - MUT vs Het:
  DEGs: {len(mut_het_sig)} (up:{(mut_het_sig['log2FoldChange']>0).sum()} down:{(mut_het_sig['log2FoldChange']<0).sum()})

PCA: PC1={pca_model.explained_variance_ratio_[0]*100:.1f}%, PC2={pca_model.explained_variance_ratio_[1]*100:.1f}%

All outputs saved to: {OUTPUT_DIR}
""")
print("Process finished successfully!")