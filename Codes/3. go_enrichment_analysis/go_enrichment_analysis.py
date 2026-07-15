#!/usr/bin/env python3
"""
3. GO Enrichment (Downstream Functional) Analysis Pipeline (bulk RNA-seq, zebrafish muscle)

PURPOSE
    Functional characterisation of the differentially expressed genes (DEGs) found by the main DE pipeline.
    It runs AFTER differential_expression_analysis.py and reads that script's CSV outputs.
    Five analyses:
        STEP 1  GO / pathway enrichment (g:Profiler) for the Q1 and Q2 up- and down-regulated gene sets, and for the Q1/Q2 overlap.
        STEP 2  Transcription-factor (TF) analysis, using the annotation 'Family' column to identify which DEGs are TFs.
        STEP 3  Gene-gene correlation analysis on the top DEGs.
        STEP 4  Detailed Q3 overlap analysis (direction breakdown, TF/biotype).
        STEP 5  Threshold-sensitivity exploration (DEG counts at stricter cut-offs).

INPUT DATASETS
    gene_count.xls             : Tab-delimited raw read-count matrix (gene_id, 20 sample columns, and gene-annotation columns including 'Family').
                                 Used for annotations and for the normalised expression underlying the correlation analysis.
    q1_genotype_DEG_results.csv: (output of the DE pipeline; Q1 genotype contrast)
    q2_age_DEG_results.csv     : (output of the DE pipeline; Q2 age contrast)

OUTPUTS (all written to OUTPUT_DIR)
    Enrichment (only written when terms are found)
        enrichment_q1_up.csv/.png, enrichment_q1_down.csv/.png, enrichment_q2_up.csv/.png, enrichment_q2_down.csv/.png, enrichment_overlap.csv/.png
    Transcription factors
        q1_transcription_factors.csv, q2_transcription_factors.csv, tf_families.png
    Correlation
        correlation_top50.png, correlation_matrix_top50.csv, correlated_gene_pairs.csv
    Overlap (Q3)
        q3_overlap_detailed.csv, overlap_fc_scatter.png
    Console
        Enrichment terms, TF summaries, correlated pairs, overlap counts, threshold tables and a final summary.


REFERENCES (this file only)
    These references apply to this source file only and are independent of any reference numbering used in the accompanying report.

    [1] U. Raudvere et al., "g:Profiler: a web server for functional enrichment analysis and conversions of gene lists
        (2019 update)", Nucleic Acids Research, vol. 47, no. W1, pp. W191-W198, 2019.
    [2] M. I. Love, W. Huber, and S. Anders, "Moderated estimation of fold change and dispersion for RNA-seq data with
        DESeq2", Genome Biology, vol. 15, no. 12, art. 550, 2014.
    [3] B. Muzellec, M. Telenczuk, V. Cabeli, and M. Andreux, "PyDESeq2: a python package for bulk RNA-seq differential
        expression analysis", Bioinformatics, vol. 39, no. 9, 2023.
    [4] M. L. Waskom, "seaborn: statistical data visualization", J. Open Source Software, vol. 6, no. 60, art. 3021, 2021.
"""

# Import all necessary libraries and modules

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') # non-interactive backend: render figures to files
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns

from gprofiler import GProfiler # g:Profiler client for enrichment [1]
from pydeseq2.preprocessing import deseq2_norm # median-of-ratios normalisation [2], [3]
from adjustText import adjust_text # non-overlapping plot labels [5]

warnings.filterwarnings('ignore') # suppress non-critical library warnings

# Configuration

# Input count matrix, the two DE result CSVs, and the output directory.
# Input: tab-delimited raw-count matrix
COUNT_MATRIX_PATH = '/Users/mehakagrawal/Desktop/Final_Dissertation/Datasets/adult_muscle_18mo/gene_count.xls'
# Input: DE pipeline output: genotype contrast
Q1_RESULTS_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/1. differential_expression_analysis/q1_genotype_DEG_results.csv'
# Input: DE pipeline output: age contrast
Q2_RESULTS_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/1. differential_expression_analysis/q2_age_DEG_results.csv'
OUTPUT_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/3. go_enrichment_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Significance basis: must match differential_expression_analysis.py (Storey q-values). SIG_COLUMN falls back to 'padj' automatically if 'qvalue' is absent.
SIG_COLUMN = 'qvalue'
SIG_CUTOFF = 0.05
LOG2FC_CUTOFF = 1.0

# Load data and define the significant gene sets

print("Loading data...")
raw_table = pd.read_csv(COUNT_MATRIX_PATH, sep='\t')
genotype_results = pd.read_csv(Q1_RESULTS_CSV, index_col=0) # Q1 (genotype)
age_results = pd.read_csv(Q2_RESULTS_CSV, index_col=0) # Q2 (age)

# Use q-values if present in both DE tables; otherwise fall back to padj.
sig_column = SIG_COLUMN if SIG_COLUMN in genotype_results.columns and SIG_COLUMN in age_results.columns else 'padj'
print(f"Significance column in use: {sig_column} (cutoff {SIG_CUTOFF}, |log2FC|>{LOG2FC_CUTOFF})")

# Rebuild the annotation table (gene_id kept as index only, to avoid duplication).
annotation_columns = ['gene_name', 'gene_chr', 'gene_start', 'gene_end', 'gene_strand', 'gene_length', 'gene_biotype', 'gene_description', 'Family']
gene_annotations = raw_table[['gene_id'] + [c for c in annotation_columns if c in raw_table.columns]].copy()
gene_annotations.set_index('gene_id', inplace=True)

# Attach readable gene names to both DE tables.
gene_name_map = gene_annotations['gene_name'].to_dict()
genotype_results['gene_name'] = genotype_results.index.map(gene_name_map)
age_results['gene_name'] = age_results.index.map(gene_name_map)

# Significant DEG sets (pass both q-value and fold-change thresholds), split by direction.
genotype_sig = genotype_results[(genotype_results[sig_column] < SIG_CUTOFF) & (genotype_results['log2FoldChange'].abs() > LOG2FC_CUTOFF)]
age_sig = age_results[(age_results[sig_column] < SIG_CUTOFF) & (age_results['log2FoldChange'].abs() > LOG2FC_CUTOFF)]

genotype_sig_up = genotype_sig[genotype_sig['log2FoldChange'] > 0]
genotype_sig_down = genotype_sig[genotype_sig['log2FoldChange'] < 0]
age_sig_up = age_sig[age_sig['log2FoldChange'] > 0]
age_sig_down = age_sig[age_sig['log2FoldChange'] < 0]

print(f"Q1 DEGs: {len(genotype_sig)} ({len(genotype_sig_up)} up, {len(genotype_sig_down)} down)")
print(f"Q2 DEGs: {len(age_sig)} ({len(age_sig_up)} up, {len(age_sig_down)} down)")

# STEP 1 - GO / pathway enrichment with g:Profiler [1]

# For each gene set ask g:Profiler which biological processes, molecular functions, cellular components (GO), and KEGG/Reactome
# pathways are over-represented, with FDR correction. Up- and down-regulated genes are tested separately because they usually represent different biology.
print("\n")
print("STEP 1: GO / Pathway Enrichment (gProfiler)")

gprofiler_client = GProfiler(return_dataframe=True)

def run_enrichment(gene_ids, label, organism='drerio'):
    """Run g:Profiler enrichment [1] for one gene list and return the results.

    Queries GO (BP/MF/CC), KEGG and Reactome for the zebrafish ('drerio') organism with FDR correction, prints the
    top 15 terms, and returns the full results DataFrame (empty on error or if nothing is significant).
    """
    print(f"\n  Running enrichment for {label} ({len(gene_ids)} genes)...")
    try:
        enrichment_result = gprofiler_client.profile(
            organism=organism,
            query=gene_ids,
            sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC'],
            significance_threshold_method='fdr',
            user_threshold=0.05,
            no_evidences=False
        )
        if len(enrichment_result) > 0:
            enrichment_result = enrichment_result.sort_values('p_value')
            print(f"  Found {len(enrichment_result)} significant terms")
            # Print the top 15 most significant terms.
            for _, row in enrichment_result.head(15).iterrows():
                source = row['source']
                term_name = row['name']
                pval = row['p_value']
                n_intersect = row['intersection_size']
                print(f"    [{source}] {term_name} (n={n_intersect}, p={pval:.2e})")
        else:
            print(f"  No significant enrichment found")
        return enrichment_result
    except Exception as error:
        print(f"  Error: {error}")
        return pd.DataFrame()

# Gene ID lists for each direction of each contrast.
genotype_up_gene_ids = genotype_sig_up.index.tolist()
genotype_down_gene_ids = genotype_sig_down.index.tolist()
age_up_gene_ids = age_sig_up.index.tolist()
age_down_gene_ids = age_sig_down.index.tolist()

enrichment_genotype_up = run_enrichment(genotype_up_gene_ids, f"Q1 Upregulated in MUT ({len(genotype_up_gene_ids)} genes)")
enrichment_genotype_down = run_enrichment(genotype_down_gene_ids, f"Q1 Downregulated in MUT ({len(genotype_down_gene_ids)} genes)")
enrichment_age_up = run_enrichment(age_up_gene_ids, f"Q2 Upregulated in Old ({len(age_up_gene_ids)} genes)")
enrichment_age_down = run_enrichment(age_down_gene_ids, f"Q2 Downregulated in Old ({len(age_down_gene_ids)} genes)")

# Enrichment on the Q1/Q2 overlap genes.
overlap_gene_ids = list(set(genotype_sig.index) & set(age_sig.index))
enrichment_overlap = run_enrichment(overlap_gene_ids, f"Q3 Overlap ({len(overlap_gene_ids)} genes)")

# Save every non-empty enrichment table.
for output_label, enrichment_df in [('q1_up', enrichment_genotype_up), ('q1_down', enrichment_genotype_down), ('q2_up', enrichment_age_up), ('q2_down', enrichment_age_down),
                                    ('overlap', enrichment_overlap)]:
    if len(enrichment_df) > 0:
        enrichment_df.to_csv(os.path.join(OUTPUT_DIR, f'enrichment_{output_label}.csv'), index=False)
        print(f"  Saved enrichment_{output_label}.csv")

# Enrichment bar plots
def plot_enrichment(enrichment_df, title, filename, top_n=15):
    """Draws and saves a horizontal bar plot of the top_n enriched terms, coloured by database source and annotated with the intersecting-gene count."""
    if len(enrichment_df) == 0:
        return
    top_terms = enrichment_df.head(top_n).copy()
    top_terms['-log10p'] = -np.log10(top_terms['p_value'].clip(lower=1e-50)) # clip avoids log(0)
    top_terms['short_name'] = top_terms['name'].str[:55]

    # One colour per annotation source.
    source_colours = {'GO:BP': '#2A9D8F', 'GO:MF': '#E9C46A', 'GO:CC': '#E76F51', 'KEGG': '#457B9D', 'REAC': '#264653'}
    bar_colours = [source_colours.get(s, '#999999') for s in top_terms['source']]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    bar_containers = ax.barh(range(len(top_terms)), top_terms['-log10p'].values, color=bar_colours, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(top_terms)))
    ax.set_yticklabels(top_terms['short_name'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('-log10(adjusted p-value)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    # Annotate each bar with its intersecting-gene count.
    for bar, n_intersect in zip(bar_containers, top_terms['intersection_size'].values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f'n={n_intersect}', va='center', fontsize=8, color='grey')

    # Legend showing only the sources actually present.
    legend_patches = [Patch(color=c, label=s) for s, c in source_colours.items() if s in top_terms['source'].values]
    if legend_patches:
        ax.legend(handles=legend_patches, fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")

plot_enrichment(enrichment_genotype_up, 'Q1: Enriched pathways - Upregulated in tert MUT', 'enrichment_q1_up.png')
plot_enrichment(enrichment_genotype_down, 'Q1: Enriched pathways - Downregulated in tert MUT', 'enrichment_q1_down.png')
plot_enrichment(enrichment_age_up, 'Q2: Enriched pathways - Upregulated in Old WT', 'enrichment_q2_up.png')
plot_enrichment(enrichment_age_down, 'Q2: Enriched pathways - Downregulated in Old WT', 'enrichment_q2_down.png')
plot_enrichment(enrichment_overlap, 'Q3: Enriched pathways - Overlap genes', 'enrichment_overlap.png')

# STEP 2 - Transcription factor analysis

# The annotation 'Family' column names the transcription-factor (TF) family of a gene (or '-' if it is not a TF).
# Flag which DEGs are TFs, summarise the families affected, and check which TFs are shared between Q1 and Q2.
print("\n")
print("STEP 2: Transcription Factor Analysis")

tf_family_col = 'Family'
if tf_family_col in gene_annotations.columns:
    # Tag each DEG with its TF family (NaN if the gene has no annotation).
    genotype_sig['tf_family'] = genotype_sig.index.map(gene_annotations[tf_family_col].to_dict())
    age_sig['tf_family'] = age_sig.index.map(gene_annotations[tf_family_col].to_dict())

    # Keep only DEGs that are annotated transcription factors.
    genotype_tfs = genotype_sig[genotype_sig['tf_family'].notna() & (genotype_sig['tf_family'] != '-') & (genotype_sig['tf_family'] != '')]
    age_tfs = age_sig[age_sig['tf_family'].notna() & (age_sig['tf_family'] != '-') & (age_sig['tf_family'] != '')]

    print(f"\nQ1 DEGs that are transcription factors: {len(genotype_tfs)} out of {len(genotype_sig)}")
    print(f"Q2 DEGs that are transcription factors: {len(age_tfs)} out of {len(age_sig)}")

    if len(genotype_tfs) > 0:
        print(f"\nQ1 - Top TF families affected by genotype:")
        genotype_tf_family_counts = genotype_tfs['tf_family'].value_counts().head(15)
        for family, count in genotype_tf_family_counts.items():
            up = (genotype_tfs[genotype_tfs['tf_family'] == family]['log2FoldChange'] > 0).sum()
            down = count - up
            print(f"  {family:30s}  total={count}  (↑{up} ↓{down})")

        print(f"\nQ1 - Top 20 DE transcription factors (by significance):")
        for _, row in genotype_tfs.sort_values(sig_column).head(20).iterrows():
            gene_display_name = row['gene_name'] if pd.notna(row['gene_name']) else row.name[:20]
            print(f"  {gene_display_name:20s}  family={row['tf_family']:20s}  log2FC={row['log2FoldChange']:+.2f}  {sig_column}={row[sig_column]:.2e}")

    if len(age_tfs) > 0:
        print(f"\nQ2 - Top TF families affected by age:")
        age_tf_family_counts = age_tfs['tf_family'].value_counts().head(15)
        for family, count in age_tf_family_counts.items():
            up = (age_tfs[age_tfs['tf_family'] == family]['log2FoldChange'] > 0).sum()
            down = count - up
            print(f"  {family:30s}  total={count}  (↑{up} ↓{down})")

        print(f"\nQ2 - Top 20 DE transcription factors (by significance):")
        for _, row in age_tfs.sort_values(sig_column).head(20).iterrows():
            gene_display_name = row['gene_name'] if pd.notna(row['gene_name']) else row.name[:20]
            print(f"  {gene_display_name:20s}  family={row['tf_family']:20s}  log2FC={row['log2FoldChange']:+.2f}  {sig_column}={row[sig_column]:.2e}")

    # Transcription factors significant in BOTH contrasts.
    shared_tfs = set(genotype_tfs.index) & set(age_tfs.index)
    if shared_tfs:
        print(f"\nTranscription factors significant in BOTH Q1 and Q2: {len(shared_tfs)}")
        for gene_id in shared_tfs:
            gene_display_name = gene_name_map.get(gene_id, gene_id[:20])
            family = gene_annotations.loc[gene_id, tf_family_col] if gene_id in gene_annotations.index else '?'
            genotype_fc = genotype_results.loc[gene_id, 'log2FoldChange']
            age_fc = age_results.loc[gene_id, 'log2FoldChange']
            direction = "SAME" if (genotype_fc > 0) == (age_fc > 0) else "OPPOSITE"
            print(f"  {gene_display_name:20s}  family={family:20s}  Q1={genotype_fc:+.2f}  Q2={age_fc:+.2f}  [{direction}]")

    # Bar plot of the TF-family distributions (bar colour = mean direction).
    if len(genotype_tfs) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        genotype_tf_summary = genotype_tfs.groupby('tf_family').agg(count=('log2FoldChange', 'size'), mean_lfc=('log2FoldChange', 'mean')).sort_values('count', ascending=False).head(12)
        genotype_bar_colours = ['#E63946' if x > 0 else '#457B9D' for x in genotype_tf_summary['mean_lfc']]
        axes[0].barh(range(len(genotype_tf_summary)), genotype_tf_summary['count'], color=genotype_bar_colours, edgecolor='white')
        axes[0].set_yticks(range(len(genotype_tf_summary)))
        axes[0].set_yticklabels(genotype_tf_summary.index, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Number of DE genes')
        axes[0].set_title('Q1 Genotype: DE Transcription Factor Families', fontweight='bold', fontsize=11)
        axes[0].grid(axis='x', alpha=0.2)

        if len(age_tfs) > 0:
            age_tf_summary = age_tfs.groupby('tf_family').agg(count=('log2FoldChange', 'size'), mean_lfc=('log2FoldChange', 'mean')).sort_values('count', ascending=False).head(12)
            age_bar_colours = ['#E63946' if x > 0 else '#457B9D' for x in age_tf_summary['mean_lfc']]
            axes[1].barh(range(len(age_tf_summary)), age_tf_summary['count'], color=age_bar_colours, edgecolor='white')
            axes[1].set_yticks(range(len(age_tf_summary)))
            axes[1].set_yticklabels(age_tf_summary.index, fontsize=9)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Number of DE genes')
            axes[1].set_title('Q2 Age: DE Transcription Factor Families', fontweight='bold', fontsize=11)
            axes[1].grid(axis='x', alpha=0.2)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'tf_families.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("\n  Saved tf_families.png")

    # Save the TF tables.
    if len(genotype_tfs) > 0:
        genotype_tfs.sort_values(sig_column).to_csv(os.path.join(OUTPUT_DIR, 'q1_transcription_factors.csv'))
        print("  Saved q1_transcription_factors.csv")
    if len(age_tfs) > 0:
        age_tfs.sort_values(sig_column).to_csv(os.path.join(OUTPUT_DIR, 'q2_transcription_factors.csv'))
        print("  Saved q2_transcription_factors.csv")
else:
    print("  No 'Family' column found in annotations - skipping TF analysis")

# STEP 3 - Gene correlation analysis on the top DEGs

# Which top DEGs move together across samples? Normalise the counts, take the 50 most significant DEGs, and compute a
# gene-gene Spearman correlation matrix, then extract strongly (co- or anti-) correlated pairs.
print("\n")
print("STEP 3: Gene Correlation Analysis")

# Rebuild the filtered, normalised expression (same filter as the DE pipeline).
sample_columns = [c for c in raw_table.columns if c.startswith('M')]
count_matrix = raw_table[['gene_id'] + sample_columns].copy().set_index('gene_id')
genes_pass_filter = (count_matrix >= 10).sum(axis=1) >= 3
filtered_counts_T = count_matrix.loc[genes_pass_filter].T.astype(int) # samples x genes

# DESeq2 median-of-ratios normalisation, then log2 [2], [3].
normalised_counts, _ = deseq2_norm(filtered_counts_T)
log_norm_expr = np.log2(normalised_counts + 1)

# Top 50 DEGs across Q1 and Q2 combined (by significance).
combined_sig = pd.concat([genotype_sig.assign(comparison='Q1'), age_sig.assign(comparison='Q2')])
top_gene_ids = combined_sig.sort_values(sig_column).head(50).index.unique().tolist()[:50]
print(f"Computing pairwise correlations for top {len(top_gene_ids)} genes...")

# Gene-gene Spearman correlation across all 20 samples.
top_gene_expr = log_norm_expr[top_gene_ids] # samples x genes
correlation_matrix = top_gene_expr.corr(method='spearman')
gene_labels = [gene_name_map.get(g, g[:12]) for g in top_gene_ids]

# Clustered heatmap of the correlation matrix [4].
cluster_grid = sns.clustermap(correlation_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1, xticklabels=gene_labels, yticklabels=gene_labels, figsize=(14, 12),
                              dendrogram_ratio=(0.12, 0.12), cbar_pos=(0.02, 0.8, 0.03, 0.12))
cluster_grid.ax_heatmap.set_xticklabels(cluster_grid.ax_heatmap.get_xticklabels(), fontsize=7, rotation=90)
cluster_grid.ax_heatmap.set_yticklabels(cluster_grid.ax_heatmap.get_yticklabels(), fontsize=7)
cluster_grid.fig.suptitle('Spearman correlation - top 50 DEGs across all samples', fontsize=13, fontweight='bold', y=1.01)
cluster_grid.savefig(os.path.join(OUTPUT_DIR, 'correlation_top50.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved correlation_top50.png")

# Save the correlation matrix (labelled by gene name).
correlation_export = correlation_matrix.copy()
correlation_export.index = gene_labels
correlation_export.columns = gene_labels
correlation_export.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix_top50.csv'))
print("  Saved correlation_matrix_top50.csv")

# Extract strongly correlated gene pairs (|Spearman r| > 0.85).
correlated_pairs = []
for i in range(len(top_gene_ids)):
    for j in range(i + 1, len(top_gene_ids)):
        r = correlation_matrix.iloc[i, j]
        if abs(r) > 0.85:
            correlated_pairs.append({
                'gene1': gene_name_map.get(top_gene_ids[i], top_gene_ids[i]),
                'gene2': gene_name_map.get(top_gene_ids[j], top_gene_ids[j]),
                'gene1_id': top_gene_ids[i],
                'gene2_id': top_gene_ids[j],
                'spearman_r': round(r, 3),
                'direction': 'co-expressed' if r > 0 else 'anti-correlated'
            })

correlated_pairs_df = pd.DataFrame(correlated_pairs).sort_values('spearman_r', key=abs, ascending=False)
print(f"\n  Strongly correlated gene pairs (|r| > 0.85): {len(correlated_pairs_df)}")
if len(correlated_pairs_df) > 0:
    print("\n  Top 20 correlated pairs:")
    for _, row in correlated_pairs_df.head(20).iterrows():
        print(f"    {row['gene1']:15s} ↔ {row['gene2']:15s}  r={row['spearman_r']:+.3f}  ({row['direction']})")
    correlated_pairs_df.to_csv(os.path.join(OUTPUT_DIR, 'correlated_gene_pairs.csv'), index=False)
    print("  Saved correlated_gene_pairs.csv")

# STEP 4 - Enhanced overlap analysis (Q3)

# For every gene significant in both contrasts, record its Q1 and Q2 fold change, whether they agree in direction, and its TF family / biotype; then visualise.
print("\n")
print("STEP 4: Enhanced Overlap Analysis (Q3)")

q3_overlap_gene_ids = list(set(genotype_sig.index) & set(age_sig.index))
print(f"\n{len(q3_overlap_gene_ids)} genes are DE in both Q1 (genotype) and Q2 (age)")

# Build the detailed overlap table.
overlap_rows = []
for gene_id in q3_overlap_gene_ids:
    genotype_fc = genotype_results.loc[gene_id, 'log2FoldChange']
    age_fc = age_results.loc[gene_id, 'log2FoldChange']
    same_direction = (genotype_fc > 0) == (age_fc > 0)
    overlap_rows.append({
        'gene_id': gene_id,
        'gene_name': gene_name_map.get(gene_id, gene_id),
        'q1_log2FC': round(genotype_fc, 3),
        'q1_sig': genotype_results.loc[gene_id, sig_column],
        'q2_log2FC': round(age_fc, 3),
        'q2_sig': age_results.loc[gene_id, sig_column],
        'direction': 'SAME' if same_direction else 'OPPOSITE',
        'tf_family': gene_annotations.loc[gene_id, 'Family'] if gene_id in gene_annotations.index and pd.notna(gene_annotations.loc[gene_id, 'Family']) else '',
        'biotype': gene_annotations.loc[gene_id, 'gene_biotype'] if gene_id in gene_annotations.index else ''
    })

overlap_detail_df = pd.DataFrame(overlap_rows).sort_values('q1_sig')
overlap_detail_df.to_csv(os.path.join(OUTPUT_DIR, 'q3_overlap_detailed.csv'), index=False)

n_same_direction = (overlap_detail_df['direction'] == 'SAME').sum()
n_opposite_direction = (overlap_detail_df['direction'] == 'OPPOSITE').sum()
print(f"  Same direction: {n_same_direction}")
print(f"  Opposite direction: {n_opposite_direction}")

# Scatter: Q1 log2FC vs Q2 log2FC for the overlap genes (colour = direction).
fig, ax = plt.subplots(figsize=(8, 7))
scatter_colours = ['#2A9D8F' if d == 'SAME' else '#E63946' for d in overlap_detail_df['direction']]
ax.scatter(overlap_detail_df['q1_log2FC'], overlap_detail_df['q2_log2FC'], c=scatter_colours, s=40, alpha=0.7, edgecolors='k', linewidth=0.3, zorder=3)

# Label transcription factors and the most extreme genes.
label_texts = []
genes_to_label = overlap_detail_df[(overlap_detail_df['tf_family'] != '') | (overlap_detail_df['q1_sig'] < 1e-10) | (overlap_detail_df['q2_sig'] < 1e-5)].head(15)
for _, row in genes_to_label.iterrows():
    label_texts.append(ax.text(row['q1_log2FC'], row['q2_log2FC'], row['gene_name'], fontsize=7, ha='center'))
if label_texts:
    adjust_text(label_texts, ax=ax, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))

ax.axhline(0, color='grey', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5)
# Reference diagonals: same direction lies along y=x, opposite along y=-x.
axis_limit = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]), abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
ax.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], '--', color='grey', alpha=0.3, linewidth=0.8)
ax.plot([-axis_limit, axis_limit], [axis_limit, -axis_limit], '--', color='grey', alpha=0.3, linewidth=0.8)

ax.set_xlabel('Q1 log2FC (Genotype: MUT vs WT)', fontsize=11)
ax.set_ylabel('Q2 log2FC (Age: Old vs Young)', fontsize=11)
ax.set_title('Overlap genes: genotype vs age fold changes', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.15)

legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#2A9D8F', markersize=8, label=f'Same direction ({n_same_direction})'),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor='#E63946', markersize=8, label=f'Opposite direction ({n_opposite_direction})')
]
ax.legend(handles=legend_handles, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overlap_fc_scatter.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved overlap_fc_scatter.png")
print("  Saved q3_overlap_detailed.csv")

# STEP 5 - Threshold-sensitivity exploration

# How many DEGs survive at progressively stricter fold-change / significance cut-offs? This shows how robust the DEG counts are to the choice of threshold.
print("\n")
print("STEP 5: Filtering Exploration")

print("\nQ1 (Genotype) - DEG counts at different thresholds:")
print(f"  {'Threshold':<25s}  {'DEGs':>6s}  {'Up':>5s}  {'Down':>5s}")
print(f"  {'-'*25}  {'-'*6}  {'-'*5}  {'-'*5}")
for lfc in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    for sig_threshold in [0.05, 0.01, 0.001]:
        mask = (genotype_results[sig_column] < sig_threshold) & (genotype_results['log2FoldChange'].abs() > lfc)
        n_deg = mask.sum()
        up = (genotype_results.loc[mask, 'log2FoldChange'] > 0).sum()
        down = n_deg - up
        if sig_threshold == 0.05:
            print(f"  |log2FC|>{lfc}, {sig_column}<{sig_threshold}  {n_deg:>6d}  {up:>5d}  {down:>5d}")

print("\nQ2 (Age) - DEG counts at different thresholds:")
print(f"  {'Threshold':<25s}  {'DEGs':>6s}  {'Up':>5s}  {'Down':>5s}")
print(f"  {'-'*25}  {'-'*6}  {'-'*5}  {'-'*5}")
for lfc in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    for sig_threshold in [0.05, 0.01, 0.001]:
        mask = (age_results[sig_column] < sig_threshold) & (age_results['log2FoldChange'].abs() > lfc)
        n_deg = mask.sum()
        up = (age_results.loc[mask, 'log2FoldChange'] > 0).sum()
        down = n_deg - up
        if sig_threshold == 0.05:
            print(f"  |log2FC|>{lfc}, {sig_column}<{sig_threshold}  {n_deg:>6d}  {up:>5d}  {down:>5d}")

# Illustrative stricter filter for a more focused gene set.
strict_genotype = genotype_results[(genotype_results[sig_column] < 0.01) & (genotype_results['log2FoldChange'].abs() > 2)]
strict_age = age_results[(age_results[sig_column] < 0.01) & (age_results['log2FoldChange'].abs() > 2)]
print(f"\nRecommended strict filter ({sig_column}<0.01, |log2FC|>2):")
print(f"  Q1: {len(strict_genotype)} genes")
print(f"  Q2: {len(strict_age)} genes")
strict_overlap_genes = set(strict_genotype.index) & set(strict_age.index)
print(f"  Overlap: {len(strict_overlap_genes)} genes")

# FINAL SUMMARY

print("\n")
print("COMPLETE SUMMARY")

print(f"""
DE Analysis:
  Q1 (Genotype: MUT vs WT at 18mo): {len(genotype_sig)} DEGs
    -> Dominated by downregulation in mutants ({(genotype_sig['log2FoldChange']<0).sum()} down vs {(genotype_sig['log2FoldChange']>0).sum()} up)
    -> Top gene by significance: {gene_name_map.get(genotype_sig.sort_values(sig_column).index[0], '?')}
  Q2 (Age: 37mo vs 11mo in WT): {len(age_sig)} DEGs
    -> {(age_sig['log2FoldChange']>0).sum()} up vs {(age_sig['log2FoldChange']<0).sum()} down in old fish
  Q3 (Overlap): {len(q3_overlap_gene_ids)} genes shared
    -> {n_opposite_direction} OPPOSITE direction ({100*n_opposite_direction/len(q3_overlap_gene_ids):.0f}%)
    -> {n_same_direction} same direction ({100*n_same_direction/len(q3_overlap_gene_ids):.0f}%)

Enrichment (gProfiler):
  -> See enrichment CSVs and plots for pathway details

Transcription Factors:
  -> Q1: {len(genotype_tfs) if 'genotype_tfs' in dir() else '?'} TFs affected by genotype
  -> Q2: {len(age_tfs) if 'age_tfs' in dir() else '?'} TFs affected by age

Correlation:
  -> {len(correlated_pairs_df) if len(correlated_pairs_df) > 0 else 0} strongly correlated gene pairs (|r|>0.85)

Output files:
  enrichment_q1_up.csv/png, enrichment_q1_down.csv/png
  enrichment_q2_up.csv/png, enrichment_q2_down.csv/png
  enrichment_overlap.csv/png
  tf_families.png, q1/q2_transcription_factors.csv
  correlation_top50.png, correlation_matrix_top50.csv
  correlated_gene_pairs.csv
  overlap_fc_scatter.png, q3_overlap_detailed.csv
""")

print("Go enrichment/KEGG analysis complete!")