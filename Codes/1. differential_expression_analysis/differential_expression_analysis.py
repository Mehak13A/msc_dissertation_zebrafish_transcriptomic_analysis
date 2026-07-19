#!/usr/bin/env python3
"""
1. Differential Expression Analysis Pipeline (bulk RNA-seq, zebrafish flank muscle)

PURPOSE
    End-to-end differential expression (DE) analysis of bulk RNA-seq read counts from zebrafish (Danio rerio)
    flank skeletal muscle. The pipeline answers three questions:
        Q1 (Genotype effect): 18-month telomerase mutant vs 18-month wild-type (M18_MUT vs M18_WT)
        Q2 (Age effect)     : 37-month wild-type vs 11-month wild-type (M37_WT vs M11_WT)
        Q3 (Overlap)        : genes significant in BOTH Q1 and Q2, and whether they move in the same or opposite direction.

INPUT DATASET
    gene_count.xls
        A tab-delimited text file (despite the .xls extension) containing raw integer read counts for every gene in every sample, plus gene-annotation columns.
        Columns:
          - gene_id : Ensemble gene identifier (index)
          - 20 sample columns (M18_MUT_1 ... M11_WT_5) : raw read counts
          - gene_name, gene_chr, gene_start, gene_end, gene_strand,
            gene_length, gene_biotype, gene_description, Family : annotations
        Sample groups: M18_MUT (n=4), M18_WT (n=5), M37_WT (n=6), M11_WT (n=5).

OUTPUTS (all written to OUTPUT_DIR)
    CSV files
        q1_genotype_DEG_results.csv: full Q1 DE table (log2FC, p, q, gene_name)
        q2_age_DEG_results.csv     : full Q2 DE table
        top50_focused_genes.csv    : 50 most significant DEGs across Q1 and Q2
    Figures (PNG)
        correlation_heatmap.png : Pearson correlation between all samples
        pca_top500.png          : PCA on the 500 most variable genes
        pca_allgenes.png        : PCA on all genes
        pca_Q1_genotype.png     : PCA restricted to the Q1 (18-month) samples
        pca_Q2_age.png          : PCA restricted to the Q2 (age) samples
        volcano_q1_genotype.png : Q1 volcano plot (fold change vs significance)
        volcano_q2_age.png      : Q2 volcano plot
        venn_overlap.png        : Q1/Q2 DEG overlap diagram
        pvalue_distributions.png: p-value histograms (model diagnostic)
        heatmap_top_degs.png    : clustered heatmap of top DEGs (z-scored)
        ma_plots.png            : MA plots (mean expression vs fold change)
    Console
        Progress messages and summary statistics for Q1, Q2 and Q3.

REFERENCES (this file only)
    These references apply to this source file only and are independent of any reference numbering used in the accompanying report.

    [1] M. I. Love, W. Huber, and S. Anders, "Moderated estimation of fold change and dispersion for RNA-seq data with
        DESeq2," Genome Biology, vol. 15, no. 12, art. 550, 2014.
    [2] B. Muzellec, M. Telenczuk, V. Cabeli, and M. Andreux, "PyDESeq2: a python package for bulk RNA-seq differential
        expression analysis," Bioinformatics, vol. 39, no. 9, 2023.
    [3] J. D. Storey and R. Tibshirani, "Statistical significance for genomewide studies," Proc. Nat. Acad. Sci.,
        vol. 100, no. 16, pp. 9440-9445, 2003.
    [4] F. Pedregosa et al., "Scikit-learn: machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.
    [5] M. L. Waskom, "seaborn: statistical data visualization," J. Open Source Software, vol. 6, no. 60, art. 3021, 2021.
"""

# Import all necessary libraries and modules

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') # non-interactive backend: render figures to files
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import seaborn as sns

from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet # DESeq2 dataset object (fits the model)
from pydeseq2.ds import DeseqStats # DESeq2 statistics object (Wald test, results)
from pydeseq2.preprocessing import deseq2_norm # median-of-ratios normalisation (for PCA/QC)
from adjustText import adjust_text # keeps volcano-plot gene labels from overlapping

warnings.filterwarnings('ignore') # suppress non-critical library warnings for a clean log

# Configuration

# Input/output locations. Alter COUNT_MATRIX_PATH accordingly based on the dataset path.
# tab-delimited raw-count matrix (see module docstring)
COUNT_MATRIX_PATH = '/Users/mehakagrawal/Desktop/Final_Dissertation/Datasets/adult_muscle_18mo/gene_count.xls'
# all CSVs and figures are written here
OUTPUT_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/1. differential_expression_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Significance thresholds. A gene is called differentially expressed only if it passes BOTH: q-value below QVALUE_CUTOFF
# (5% false discovery rate) AND an absolute log2 fold change above LOG2FC_CUTOFF (at least a two-fold change).
QVALUE_CUTOFF = 0.05
LOG2FC_CUTOFF = 1.0

# Number of most-variable genes used for the primary PCA and correlation views.
N_TOP_VARIABLE_GENES = 500

# Storey q-value helper

def storey_qvalues(pvals):
    """Compute Storey & Tibshirani (2003) q-values from a vector of p-values.

    A q-value is the minimum false discovery rate (FDR) at which a given test can be called significant. It equals
    pi0 * (Benjamini-Hochberg-adjusted p-value), where pi0 is the estimated proportion of genes that are truly null
    (not DE). Because pi0 <= 1, q-values are never larger than BH p-values, giving a slightly more powerful but still
    FDR-controlled cut-off. The RANKING of genes is unchanged relative to BH.

    ATTRIBUTION: implemented from scratch following the q-value procedure of Storey & Tibshirani [3].
    The smoothing-spline estimate of pi0 follows that paper.

    Parameters
    pvals : array-like
        Raw p-values (may contain NaN for genes that were not tested).

    Returns
    tuple(np.ndarray, float)
        q-values aligned to the input order (NaN kept where the input was NaN), and the estimated null proportion pi0.
    """
    pvals_arr = np.asarray(pvals, dtype=float)
    valid_mask = ~np.isnan(pvals_arr) # ignore genes with no p-value
    valid_pvals = pvals_arr[valid_mask]
    n_valid = len(valid_pvals)
    if n_valid == 0:
        return np.full_like(pvals_arr, np.nan), np.nan # returns NaN placeholders

    # Estimate pi0 (fraction of true nulls) across a grid of lambda thresholds.
    lambdas = np.arange(0.05, 0.96, 0.05)
    pi0s = [np.mean(valid_pvals > lam) / (1.0 - lam) for lam in lambdas]
    try:
        # Smoothing-spline estimate of pi0 evaluated at the largest lambda, as in Storey & Tibshirani (2003).
        # Imported here so the function still runs (falling back below) if SciPy is unavailable.
        from scipy.interpolate import UnivariateSpline
        pi0 = float(UnivariateSpline(lambdas, pi0s, k=3, s=0)(lambdas[-1]))
    except Exception:
        pi0 = pi0s[-1]
    pi0 = min(max(pi0, 1e-8), 1.0) # clamp to a sensible (0, 1] range

    # Convert sorted p-values to q-values, then enforce monotonicity.
    ascending_order = np.argsort(valid_pvals)
    pvals_sorted = valid_pvals[ascending_order]
    qvals_sorted = pi0 * n_valid * pvals_sorted / np.arange(1, n_valid + 1)
    qvals_sorted = np.minimum.accumulate(qvals_sorted[::-1])[::-1] # q must be non-decreasing in p
    qvals_sorted = np.minimum(qvals_sorted, 1.0)

    # Scatter q-values back to the original gene order.
    qvals_ranked = np.empty(n_valid)
    qvals_ranked[ascending_order] = qvals_sorted
    qvals_full = np.full_like(pvals_arr, np.nan)
    qvals_full[valid_mask] = qvals_ranked
    return qvals_full, pi0

# STEP 1 - Load and prepare data

# gene_count.xls is tab-delimited (despite the .xls extension). Split it into a count matrix (the 20 sample columns of
# raw integer read counts) and a table of gene annotations (name, chromosome, biotype, etc.).
# Raw counts = the number of sequenced RNA fragments mapped to each gene in each sample.
print("STEP 1: Loading and preparing data")

raw_table = pd.read_csv(COUNT_MATRIX_PATH, sep='\t')

# Sample columns all start with 'M' (e.g. M18_MUT_1); everything else is annotation.
sample_columns = [c for c in raw_table.columns if c.startswith('M')]
# gene_id is handled separately as the index (kept out of annotation_columns to avoid a duplicate-column bug when it is also used as the index).
annotation_columns = ['gene_name', 'gene_chr', 'gene_start', 'gene_end', 'gene_strand', 'gene_length', 'gene_biotype', 'gene_description', 'Family']

# Annotation table, indexed by gene_id.
gene_annotations = raw_table[['gene_id'] + [c for c in annotation_columns if c in raw_table.columns]].copy()
gene_annotations.set_index('gene_id', inplace=True)

# Count matrix (genes x samples), indexed by gene_id.
count_matrix = raw_table[['gene_id'] + sample_columns].copy()
count_matrix.set_index('gene_id', inplace=True)

print(f"Raw count matrix: {count_matrix.shape[0]} genes × {count_matrix.shape[1]} samples")
print(f"\nSample groups:")
for grp in ['M18_MUT', 'M18_WT', 'M37_WT', 'M11_WT']:
    n = sum(1 for c in sample_columns if c.startswith(grp))
    print(f"  {grp}: n={n}")

# Lookup from gene_id -> readable gene name, with a fallback for unnamed genes.
gene_name_map = gene_annotations['gene_name'].to_dict()

def gene_label(gene_id):
    """Return the readable gene name for a gene_id, or a truncated Ensembl ID if no name is available (used for plot and console labels)."""
    gene_name_val = gene_name_map.get(gene_id)
    if gene_name_val and pd.notna(gene_name_val) and gene_name_val.strip():
        return gene_name_val
    return gene_id[:18]

# STEP 1b - Build sample metadata

# DESeq2 needs a table telling it which experimental group each sample belongs to. The sample name encodes everything:
# e.g. M18_MUT_1 = 18-month, mutant, rep 1. We derive: group (e.g. M18_MUT), genotype (MUT/WT) and age (11mo/18mo/37mo).
sample_metadata = pd.DataFrame(index=sample_columns)
sample_metadata['group'] = [c.rsplit('_', 1)[0] for c in sample_columns]
sample_metadata['genotype'] = ['MUT' if 'MUT' in c else 'WT' for c in sample_columns]
sample_metadata['age'] = ['18mo' if 'M18' in c else '37mo' if 'M37' in c else '11mo' for c in sample_columns]

print(f"\nMetadata:\n{sample_metadata}")

# STEP 2 - Pre-filter low-count genes

# Remove genes that are essentially unexpressed: keep a gene only if it has at least MIN_READS_PER_GENE reads in at
# least MIN_SAMPLES_EXPRESSING samples. Very low-count genes add noise and reduce statistical power.
print("\n")
print("STEP 2: Pre-filtering low-count genes")

min_reads_per_gene = 10
min_samples_expressing = 3
# Filtering: A gene is kept only if it has ≥10 reads in at least 3 samples.
genes_pass_filter = (count_matrix >= min_reads_per_gene).sum(axis=1) >= min_samples_expressing
filtered_counts = count_matrix.loc[genes_pass_filter]
print(f"Genes before filtering: {count_matrix.shape[0]}")
print(f"Genes after filtering (>= {min_reads_per_gene} reads in >= {min_samples_expressing} samples): {filtered_counts.shape[0]}")
print(f"Genes removed: {count_matrix.shape[0] - filtered_counts.shape[0]}")

# STEP 3 - Library size summary

# Library size = total mapped reads per sample. Large, consistent library sizes indicate no sample is drastically under-sequenced.
# DESeq2 normalisation below corrects for the remaining differences in sequencing depth.
print("\n")
print("STEP 3: Library size summary")

library_sizes = filtered_counts.sum(axis=0)
print("\nLibrary sizes (millions of reads):")
for sample_name, size in library_sizes.items():
    print(f"  {sample_name}: {size/1e6:.1f}M")

# STEP 4 - Normalisation, correlation heatmap and PCA

# Normalise counts with the DESeq2 median-of-ratios method (deseq2_norm), log2-transform them, then explore sample
# structure with a correlation heatmap and PCA. PCA finds the axes of greatest variation, showing which samples group together.
print("\n")
print("STEP 4: Running DESeq2 normalisation + PCA")

filtered_counts_T = filtered_counts.T.astype(int) # DESeq2 expects samples x genes

# Median-of-ratios normalisation (DESeq2 [1], via the pydeseq2 implementation [2]).
normalised_counts, size_factors = deseq2_norm(filtered_counts_T)
log_norm_by_sample = np.log2(normalised_counts + 1) # samples x genes (for PCA)
log_norm_by_gene = log_norm_by_sample.T # genes x samples (for sample correlation)

# Fixed colour per experimental group (ggplot2 default hues) for all plots.
group_colours = {'M18_MUT': '#F8766D', 'M18_WT': '#7CAE00', 'M37_WT': '#00BFC4', 'M11_WT': '#C77CFF'}

# Correlation heatmap (Pearson correlation between samples) computes Pearson correlation between every pair of samples'
# expression profiles. Samples ordered by group so within-group similarity is visible as blocks.
heatmap_sample_order = ([c for c in sample_columns if c.startswith('M18_MUT')] + [c for c in sample_columns if c.startswith('M18_WT')] +
                        [c for c in sample_columns if c.startswith('M37_WT')] + [c for c in sample_columns if c.startswith('M11_WT')])
sample_correlation = log_norm_by_gene.corr(method='pearson').loc[heatmap_sample_order, heatmap_sample_order]

# Custom white-to-blue colour map for the correlation values.
correlation_cmap = LinearSegmentedColormap.from_list('rb', ['#FFFFFF', '#B9B9F0', '#3C3CE6', '#0000CD'])
fig, ax = plt.subplots(figsize=(12, 10))
correlation_display = sample_correlation.values[::-1] # flip rows so the diagonal reads top-left to bottom-right
heatmap_image = ax.imshow(correlation_display, cmap=correlation_cmap, vmin=sample_correlation.values[sample_correlation.values < 1].min(), vmax=1.0)
ax.set_xticks(range(len(heatmap_sample_order)))
ax.set_yticks(range(len(heatmap_sample_order)))
ax.set_xticklabels(heatmap_sample_order, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(heatmap_sample_order[::-1], fontsize=9)
# Annotate each cell with its correlation value (white text on dark cells).
for i in range(len(heatmap_sample_order)):
    for j in range(len(heatmap_sample_order)):
        value = correlation_display[i, j]
        ax.text(j, i, f'{value:.3g}', ha='center', va='center', fontsize=6.5, color='white' if value > 0.8 else 'black')
ax.set_title('Pearson correlation between samples', fontsize=16, pad=15)
fig.colorbar(heatmap_image, ax=ax, shrink=0.5, label='R')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

# PCA helper: labelled PC1 vs PC2 scatter for a set of samples
def generate_pca_plot(expr_by_gene, output_filename, plot_title, n_top_genes=None, sample_subset=None):
    """Runs PCA and saves a labelled PC1-vs-PC2 scatter plot.

    Parameters
    expr_by_gene : DataFrame
        Log-normalised expression, genes x samples.
    output_filename : str
        File name (within OUTPUT_DIR) for the saved figure.
    plot_title : str
        Title shown on the plot.
    n_top_genes : int or None
        If given, restrict PCA to this many most-variable genes.
    sample_subset : list or None
        If given, restrict PCA to these sample columns.
    """
    expr_subset = expr_by_gene if sample_subset is None else expr_by_gene[sample_subset]
    if n_top_genes:
        # Keep only the most variable genes as they carry the biological signal.
        gene_variances = expr_subset.var(axis=1).sort_values(ascending=False)
        expr_subset = expr_subset.loc[gene_variances.head(n_top_genes).index]

    expr_by_sample = expr_subset.T # PCA expects samples as rows
    pca_model = PCA(n_components=3).fit(expr_by_sample)
    pc_coords = pd.DataFrame(pca_model.transform(expr_by_sample), index=expr_by_sample.index, columns=['PC1', 'PC2', 'PC3'])
    variance_explained = pca_model.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.axhline(0, color='grey', ls='-.', lw=1, alpha=0.6)
    ax.axvline(0, color='grey', ls='-.', lw=1, alpha=0.6)
    # Plot each group in its own colour.
    groups_in_plot = [g for g in ['M18_MUT', 'M18_WT', 'M37_WT', 'M11_WT'] if any(s.rsplit('_', 1)[0] == g for s in expr_by_sample.index)]
    for grp in groups_in_plot:
        group_member_samples = [s for s in expr_by_sample.index if s.rsplit('_', 1)[0] == grp]
        ax.scatter(pc_coords.loc[group_member_samples, 'PC1'], pc_coords.loc[group_member_samples, 'PC2'], c=group_colours[grp], s=120, label=grp, edgecolors='none', zorder=3)
    # Label every point with its sample name.
    for s in expr_by_sample.index:
        ax.annotate(s, (pc_coords.loc[s, 'PC1'], pc_coords.loc[s, 'PC2']), fontsize=8, color=group_colours[s.rsplit('_', 1)[0]], xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel(f'PC1 ({variance_explained[0]:.2f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({variance_explained[1]:.2f}%)', fontsize=12)
    ax.set_title(plot_title, fontsize=12)
    ax.legend(title='group', loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_filename}  (PC1={variance_explained[0]:.1f}%, PC2={variance_explained[1]:.1f}%)")

# Sample subsets used for the per-question PCAs.
genotype_pca_samples = [c for c in sample_columns if c.startswith('M18')] # Q1 samples
age_pca_samples = [c for c in sample_columns if c.startswith('M37') or c.startswith('M11')] # Q2 samples

generate_pca_plot(log_norm_by_gene, 'pca_top500.png', f'PCA (top {N_TOP_VARIABLE_GENES} variable genes)', n_top_genes=N_TOP_VARIABLE_GENES)
generate_pca_plot(log_norm_by_gene, 'pca_allgenes.png', 'PCA (all genes)')
generate_pca_plot(log_norm_by_gene, 'pca_Q1_genotype.png', 'PCA - Q1 samples (M18_MUT vs M18_WT)', sample_subset=genotype_pca_samples)
generate_pca_plot(log_norm_by_gene, 'pca_Q2_age.png', 'PCA - Q2 samples (M37_WT vs M11_WT)', sample_subset=age_pca_samples)

# STEP 5 - DE analysis, Q1: Genotype effect (M18_MUT vs M18_WT)

# Core analysis. Using only the nine 18-month samples (4 MUT + 5 WT) with a design of ~genotype (the only variable of interest),
# DESeq2 fits a negative-binomial model per gene, estimates dispersion with Empirical Bayes shrinkage (borrowing information
# across genes, important when n is small), and runs a Wald test. The Wald test checks whether an estimated effect (log2 fold change)
# is significantly different from zero by comparing it to its own standard error, essentially asking "is this effect large relative to how uncertain we are about it?"
# Output: a log2 fold change and a p-value per gene. Convert p-values to Storey q-values.
print("\n")
print("STEP 5: DE Analysis - Q1: Genotype effect (M18_MUT vs M18_WT)")

genotype_samples = [c for c in sample_columns if c.startswith('M18')]
genotype_counts = filtered_counts[genotype_samples].T.astype(int) # samples x genes
genotype_metadata = sample_metadata.loc[genotype_samples].copy()

dds_genotype = DeseqDataSet(counts=genotype_counts, metadata=genotype_metadata, design="~genotype")
dds_genotype.deseq2()

# Contrast: MUT relative to WT (positive log2FC -> higher in mutant).
stats_genotype = DeseqStats(dds_genotype, contrast=["genotype", "MUT", "WT"])
stats_genotype.summary()

genotype_results = stats_genotype.results_df.copy()
genotype_results['gene_name'] = genotype_results.index.map(gene_name_map)
genotype_results['qvalue'], genotype_pi0 = storey_qvalues(genotype_results['pvalue'])
genotype_results.to_csv(os.path.join(OUTPUT_DIR, 'q1_genotype_DEG_results.csv'))

# Keep only genes passing both the q-value and fold-change thresholds.
genotype_significant = genotype_results[(genotype_results['qvalue'] < QVALUE_CUTOFF) & (genotype_results['log2FoldChange'].abs() > LOG2FC_CUTOFF)]
print(f"\nQ1 significant DEGs (q<{QVALUE_CUTOFF}, |log2FC|>{LOG2FC_CUTOFF}): {len(genotype_significant)}   [pi0={genotype_pi0:.3f}]")
print(f"  Upregulated in MUT: {(genotype_significant['log2FoldChange'] > 0).sum()}")
print(f"  Downregulated in MUT: {(genotype_significant['log2FoldChange'] < 0).sum()}")

print("\nTop 20 DEGs by q-value (Q1 - Genotype):")
genotype_top20 = genotype_significant.sort_values('qvalue').head(20)
for _, row in genotype_top20.iterrows():
    print(f"  {gene_label(row.name):20s}  log2FC={row['log2FoldChange']:+.2f}  qvalue={row['qvalue']:.2e}")

# STEP 6 - DE analysis, Q2: Age effect (M37_WT vs M11_WT)

# Same procedure on the 11 wild-type samples only (6 old M37_WT + 5 young M11_WT) with design ~age_group.
# Holding genotype constant (all WT) isolates the effect of natural ageing.
print("\n")
print("STEP 6: DE Analysis - Q2: Age effect (M37_WT vs M11_WT)")

age_samples = [c for c in sample_columns if c.startswith('M37') or c.startswith('M11')]
age_counts = filtered_counts[age_samples].T.astype(int)
age_metadata = sample_metadata.loc[age_samples].copy()
age_metadata['age_group'] = ['old' if 'M37' in s else 'young' for s in age_samples]

dds_age = DeseqDataSet(counts=age_counts, metadata=age_metadata, design="~age_group")
dds_age.deseq2()

# Contrast: old relative to young (positive log2FC -> higher in old fish).
stats_age = DeseqStats(dds_age, contrast=["age_group", "old", "young"])
stats_age.summary()

age_results = stats_age.results_df.copy()
age_results['gene_name'] = age_results.index.map(gene_name_map)
age_results['qvalue'], age_pi0 = storey_qvalues(age_results['pvalue'])
age_results.to_csv(os.path.join(OUTPUT_DIR, 'q2_age_DEG_results.csv'))

age_significant = age_results[(age_results['qvalue'] < QVALUE_CUTOFF) & (age_results['log2FoldChange'].abs() > LOG2FC_CUTOFF)]
print(f"\nQ2 significant DEGs (q<{QVALUE_CUTOFF}, |log2FC|>{LOG2FC_CUTOFF}): {len(age_significant)}   [pi0={age_pi0:.3f}]")
print(f"  Upregulated in old: {(age_significant['log2FoldChange'] > 0).sum()}")
print(f"  Downregulated in old: {(age_significant['log2FoldChange'] < 0).sum()}")

print("\nTop 20 DEGs by q-value (Q2 - Age):")
age_top20 = age_significant.sort_values('qvalue').head(20)
for _, row in age_top20.iterrows():
    print(f"  {gene_label(row.name):20s}  log2FC={row['log2FoldChange']:+.2f}  qvalue={row['qvalue']:.2e}")

# STEP 7 - Overlap between Q1 and Q2 (Q3)

# Intersect the two significant gene sets. For each shared gene, compare the sign of its fold change in Q1 vs Q2 (same direction or opposite).
print("\n")
print("STEP 7: Overlap between Q1 and Q2 DEGs (Q3)")

genotype_gene_set = set(genotype_significant.index)
age_gene_set = set(age_significant.index)
shared_genes = genotype_gene_set & age_gene_set

print(f"Q1 DEGs: {len(genotype_gene_set)}")
print(f"Q2 DEGs: {len(age_gene_set)}")
print(f"Overlap: {len(shared_genes)}")

if shared_genes:
    print("\nOverlapping genes:")
    for gene_id in shared_genes:
        fc_genotype = genotype_results.loc[gene_id, 'log2FoldChange']
        fc_age = age_results.loc[gene_id, 'log2FoldChange']
        direction = "SAME" if (fc_genotype > 0) == (fc_age > 0) else "OPPOSITE"
        print(f"  {gene_label(gene_id):20s}  Q1 log2FC={fc_genotype:+.2f}  Q2 log2FC={fc_age:+.2f}  [{direction}]")

# STEP 8 - Volcano plots

# A volcano plot shows fold change (x) vs statistical significance (y), making strongly and significantly changed genes stand out in the upper corners.
print("\n")
print("STEP 8: Generating volcano plots")

def generate_volcano_plot(de_results, plot_title, output_path, top_n=15):
    """Draws and saves a volcano plot (log2 fold change vs -log10 q-value).

    Significant up/down genes are coloured; the top_n most significant are labelled. Label positions are de-overlapped with the adjustText library.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    plot_data = de_results.dropna(subset=['qvalue', 'log2FoldChange']).copy()
    plot_data['-log10q'] = -np.log10(plot_data['qvalue'].clip(lower=1e-300)) # clip avoids log(0)

    # Classify points by significance and direction.
    significant_up = (plot_data['qvalue'] < QVALUE_CUTOFF) & (plot_data['log2FoldChange'] > LOG2FC_CUTOFF)
    significant_down = (plot_data['qvalue'] < QVALUE_CUTOFF) & (plot_data['log2FoldChange'] < -LOG2FC_CUTOFF)
    not_significant = ~(significant_up | significant_down)

    ax.scatter(plot_data.loc[not_significant, 'log2FoldChange'], plot_data.loc[not_significant, '-log10q'], c='#CCCCCC', s=8, alpha=0.5, label='NS', zorder=1)
    ax.scatter(plot_data.loc[significant_up, 'log2FoldChange'], plot_data.loc[significant_up, '-log10q'], c='#E63946', s=20, alpha=0.7, label=f'Up ({significant_up.sum()})', zorder=2)
    ax.scatter(plot_data.loc[significant_down, 'log2FoldChange'], plot_data.loc[significant_down, '-log10q'], c='#457B9D', s=20, alpha=0.7, label=f'Down ({significant_down.sum()})', zorder=2)

    # Label the most significant genes.
    genes_to_label = plot_data[significant_up | significant_down].nlargest(top_n, '-log10q')
    label_texts = []
    for idx, row in genes_to_label.iterrows():
        label_texts.append(ax.text(row['log2FoldChange'], row['-log10q'], gene_label(idx), fontsize=7, ha='center', va='bottom'))
    if label_texts:
        # adjustText library: repositions labels to avoid overlap.
        adjust_text(label_texts, ax=ax, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))

    # Threshold guide lines.
    ax.axhline(-np.log10(QVALUE_CUTOFF), color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(LOG2FC_CUTOFF, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(-LOG2FC_CUTOFF, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlabel('log2 Fold Change', fontsize=12)
    ax.set_ylabel('-log10 q-value', fontsize=12)
    ax.set_title(plot_title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

generate_volcano_plot(genotype_results, 'Q1: Genotype Effect - M18_MUT vs M18_WT', os.path.join(OUTPUT_DIR, 'volcano_q1_genotype.png'))
generate_volcano_plot(age_results, 'Q2: Age Effect - M37_WT (old) vs M11_WT (young)', os.path.join(OUTPUT_DIR, 'volcano_q2_age.png'))

# STEP 8b - Venn-style overlap diagram

# Two overlapping circles showing the number of DEGs unique to Q1, unique to Q2, and shared between them.
fig, ax = plt.subplots(figsize=(6, 5))

circle_genotype = Circle((-0.3, 0), 1.0, alpha=0.3, color='#E63946', label=f'Q1 Genotype ({len(genotype_gene_set)})')
circle_age = Circle((0.3, 0), 1.0, alpha=0.3, color='#457B9D', label=f'Q2 Age ({len(age_gene_set)})')
ax.add_patch(circle_genotype)
ax.add_patch(circle_age)
ax.text(-0.8, 0, f'{len(genotype_gene_set - age_gene_set)}', fontsize=18, ha='center', va='center', fontweight='bold')
ax.text(0.0, 0, f'{len(shared_genes)}', fontsize=18, ha='center', va='center', fontweight='bold')
ax.text(0.8, 0, f'{len(age_gene_set - genotype_gene_set)}', fontsize=18, ha='center', va='center', fontweight='bold')
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.legend(fontsize=10, loc='upper center')
ax.set_title('DEG Overlap: Genotype vs Age', fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'venn_overlap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: venn_overlap.png")

# STEP 8c - p-value distribution diagnostic

# Histograms of raw p-values. A well-behaved DE test shows a roughly uniform distribution with a peak near zero (the true positives).
# This is a standard sanity check on the model fit.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(genotype_results['pvalue'].dropna(), bins=50, color='#E63946', alpha=0.7, edgecolor='white')
axes[0].set_title('Q1 Genotype: p-value distribution')
axes[0].set_xlabel('p-value')
axes[0].set_ylabel('Frequency')
axes[1].hist(age_results['pvalue'].dropna(), bins=50, color='#457B9D', alpha=0.7, edgecolor='white')
axes[1].set_title('Q2 Age: p-value distribution')
axes[1].set_xlabel('p-value')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pvalue_distributions.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: pvalue_distributions.png")

# STEP 9 - Clustered heatmap of top DEGs

# Z-scored, log-normalised expression of the union of the top-25 Q1 and top-25 Q2 DEGs (capped at 40 genes) across
# all samples, with hierarchical clustering on both rows and columns (seaborn clustermap).
print("\n")
print("STEP 9: Top gene heatmap")

top_genotype_degs = genotype_significant.nsmallest(25, 'qvalue').index.tolist()
top_age_degs = age_significant.nsmallest(25, 'qvalue').index.tolist()
top_degs_for_heatmap = list(dict.fromkeys(top_genotype_degs + top_age_degs))[:40] # de-duplicate, cap at 40

heatmap_expr = log_norm_by_sample[top_degs_for_heatmap].T
# Z-score each gene across samples (mean 0, sd 1) so colours are comparable per row.
heatmap_zscores = (heatmap_expr - heatmap_expr.mean(axis=1).values[:, None]) / heatmap_expr.std(axis=1).values[:, None]

heatmap_row_labels = [gene_label(g) for g in top_degs_for_heatmap]
heatmap_col_colours = [group_colours[sample_metadata.loc[s, 'group']] for s in sample_columns]

cluster_grid = sns.clustermap(heatmap_zscores, cmap='RdBu_r', center=0, yticklabels=heatmap_row_labels, xticklabels=[s.replace('_', '\n') for s in sample_columns],
                              col_colors=heatmap_col_colours, figsize=(14, max(8, len(top_degs_for_heatmap) * 0.3)), dendrogram_ratio=(0.1, 0.15),
                              cbar_pos=(0.02, 0.8, 0.03, 0.15), row_cluster=True, col_cluster=True)
cluster_grid.ax_heatmap.set_ylabel('')
cluster_grid.fig.suptitle('Top DEGs: Z-scored normalised expression', fontsize=13, fontweight='bold', y=1.01)
cluster_grid.savefig(os.path.join(OUTPUT_DIR, 'heatmap_top_degs.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: heatmap_top_degs.png")

# STEP 10 - Summary statistics

print("\n")
print("SUMMARY")
print(f"Total genes analysed: {filtered_counts.shape[0]}")

print(f"\nQ1 (Genotype: M18_MUT vs M18_WT):")
print(f"  Significant DEGs: {len(genotype_significant)} (q<{QVALUE_CUTOFF}, |log2FC|>{LOG2FC_CUTOFF})")
print(f"  Up in MUT: {(genotype_significant['log2FoldChange']>0).sum()}, Down in MUT: {(genotype_significant['log2FoldChange']<0).sum()}")
if len(genotype_top20) > 0:
    print(f"  Most significant: {gene_label(genotype_top20.index[0])} (qvalue={genotype_top20.iloc[0]['qvalue']:.2e})")

print(f"\nQ2 (Age: M37_WT vs M11_WT):")
print(f"  Significant DEGs: {len(age_significant)} (q<{QVALUE_CUTOFF}, |log2FC|>{LOG2FC_CUTOFF})")
print(f"  Up in old: {(age_significant['log2FoldChange']>0).sum()}, Down in old: {(age_significant['log2FoldChange']<0).sum()}")
if len(age_top20) > 0:
    print(f"  Most significant: {gene_label(age_top20.index[0])} (qvalue={age_top20.iloc[0]['qvalue']:.2e})")

print(f"\nQ3 (Overlap): {len(shared_genes)} genes shared between Q1 and Q2")

# STEP 11 - Export focused gene list

# Combine the Q1 and Q2 significant tables, tag each with its comparison, and save the 50 most significant genes overall for downstream work.
combined_significant = pd.concat([genotype_significant.assign(comparison='Q1_genotype'), age_significant.assign(comparison='Q2_age')])
combined_significant_sorted = combined_significant.sort_values('qvalue')
top50_focused = combined_significant_sorted.head(50)
top50_focused.to_csv(os.path.join(OUTPUT_DIR, 'top50_focused_genes.csv'))
print(f"\nTop 50 focused gene list saved (for next steps: correlation, enrichment)")

# STEP 12 - MA plots

# An MA plot shows mean expression (x) vs log2 fold change (y). It checks whether differential expression is biased toward high- or low-expression genes.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ma_results, title in [(axes[0], genotype_results, 'Q1: Genotype (MUT vs WT)'), (axes[1], age_results, 'Q2: Age (Old vs Young)')]:
    ma_data = ma_results.dropna(subset=['qvalue', 'log2FoldChange', 'baseMean']).copy()
    ma_significant = (ma_data['qvalue'] < QVALUE_CUTOFF) & (ma_data['log2FoldChange'].abs() > LOG2FC_CUTOFF)
    ax.scatter(np.log10(ma_data.loc[~ma_significant, 'baseMean']+1), ma_data.loc[~ma_significant, 'log2FoldChange'], c='#CCCCCC', s=5, alpha=0.4)
    ax.scatter(np.log10(ma_data.loc[ma_significant, 'baseMean']+1), ma_data.loc[ma_significant, 'log2FoldChange'], c='#E63946', s=10, alpha=0.6)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.axhline(1, color='grey', linestyle='--', linewidth=0.5)
    ax.axhline(-1, color='grey', linestyle='--', linewidth=0.5)
    ax.set_xlabel('log10(mean expression + 1)')
    ax.set_ylabel('log2 Fold Change')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ma_plots.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: ma_plots.png")

print("\n Pipeline complete! All outputs saved.")
