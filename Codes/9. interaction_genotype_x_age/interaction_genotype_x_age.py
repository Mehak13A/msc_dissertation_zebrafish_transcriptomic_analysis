#!/usr/bin/env python3
"""
9. Genotype x Age Interaction Model with Per-Gene Classification

PURPOSE
    Tests whether the tert-/- genotype effect depends on Age by combining the 3-month (JenAge) and 18-month (KCL adult)
    datasets into a 2x2 design: genotype (WT / MUT)  x  age (3mo / 18mo)

    The interaction term genotype:age is the formal version of the project's central question: is telomerase loss just
    accelerated ageing, or does it do something different with age? It is fitted as a fixed-effects GLM in DESeq2 (design ~genotype + age + genotype:age).

    On top of the interaction contrast, the program extracts every biologically meaningful contrast from the same
    fitted model and builds one per-gene classification table: the genotype effect at each age, the age effect in each
    genotype, the interaction, the four group means, and category labels that say how each gene behaves (affected by
    genotype, by age, by both, or neither, and how the genotype effect changes across age).

INPUTS
    INPUT_18M_XLS: the 18-month adult-muscle raw count matrix (gene_count.xls), tab separated, with a gene_id column,
                   a gene_name column, and sample columns including M18_MUT* and M18_WT*.
    INPUT_3M_CSV : the 3-month raw count matrix (tert3m_raw_counts.csv), gene id as index, with M3_MUT* and M3_WT* sample columns (Het is dropped).

OUTPUTS (all written to OUTPUT_DIR)
    interaction_genotype_x_age_results.csv: the interaction contrast per gene (interaction_log2FC, qvalue, gene_name).
    interaction_gene_classification.csv   : the full per-gene table: four group means and SEMs, the genotype effect at 3mo and 18mo,
                                            the age effect in WT and MUT, the interaction, significance flags, and the category labels.
    interaction_category_summary.csv      : gene counts per category.
    interaction_lineplots.png             : trajectory panels (WT and MUT means with error bars, 3mo to 18mo) for the top interaction genes and tp53.
    interaction_pattern_counts.png        : bar charts of the category counts.
    interaction_geno_shift_scatter.png    : genotype effect at 3mo versus 18mo, coloured by interaction pattern.

References (this file only)
    These references apply to this source file only and are independent of any reference numbering used in the accompanying report.

    [1] M. I. Love, W. Huber, and S. Anders, "Moderated estimation of fold change and dispersion for RNA-seq data with
        DESeq2," Genome Biology, vol. 15, no. 12, art. 550, 2014.
    [2] B. Muzellec, M. Telenczuk, V. Cabeli, and M. Andreux, "PyDESeq2: a python package for bulk RNA-seq differential
        expression analysis," Bioinformatics, vol. 39, no. 9, 2023.
    [3] J. D. Storey and R. Tibshirani, "Statistical significance for genomewide studies," Proc. Nat. Acad. Sci., vol. 100, no. 16, pp. 9440-9445, 2003.
"""

# Import all the necessary libraries and modules

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # non-interactive backend: render figures to files
import matplotlib.pyplot as plt
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

warnings.filterwarnings('ignore')

# CONFIGURATION (input and output paths as per my computer)
INPUT_18M_XLS = '/Users/mehakagrawal/Desktop/Final_Dissertation/Datasets/adult_muscle_18mo/gene_count.xls'
INPUT_3M_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/4. tert_3month_de_analysis/tert3m_raw_counts.csv'
OUTPUT_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/9. interaction_genotype_x_age'

QVALUE_CUT = 0.05 # significance threshold for every effect flag
LOG2FC_CUT = 1.0 # magnitude threshold used only when selecting genes to plot
MIN_COUNT = 10 # a gene must reach this count
MIN_SAMPLES = 3 # in at least this many samples to be kept

def storey_qvalues(pvals):
    """Storey and Tibshirani q-values from a vector of p-values."""
    p = np.asarray(pvals, float)
    mask = ~np.isnan(p) # keep track of which entries are valid (non-NaN) p-values
    pv = p[mask] # work only with the valid p-values from here on
    m = len(pv)
    if m == 0:
        return np.full_like(p, np.nan), np.nan
    lambdas = np.arange(0.05, 0.96, 0.05)
    pi0s = [np.mean(pv > lam) / (1.0 - lam) for lam in lambdas]
    try:
        # Fit a smooth spline through the (lambda, pi0) estimates and take its value at the largest lambda, following Storey & Tibshirani's smoothing approach
        from scipy.interpolate import UnivariateSpline
        pi0 = float(UnivariateSpline(lambdas, pi0s, k=3, s=0)(lambdas[-1]))
    except Exception:
        pi0 = pi0s[-1]
    pi0 = min(max(pi0, 1e-8), 1.0)
    order = np.argsort(pv)
    sorted_p = pv[order]
    # Raw q-value estimate at each rank: pi0 * m * p(i) / rank(i) (this is the FDR-style scaling, using the estimated pi0 instead of assuming pi0 = 1)
    q = pi0 * m * sorted_p / np.arange(1, m + 1)
    # Enforce monotonicity: q-values must not decrease as p-values increase, so take the running minimum from the largest p-value down to the smallest
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    ordered_q = np.empty(m)
    ordered_q[order] = q
    out = np.full_like(p, np.nan)
    out[mask] = ordered_q
    return out, pi0

def load_counts(path_18m, path_3m):
    """Load and combine the 18-month and 3-month count matrices on shared genes."""
    # 18-month: drop annotation columns, keep the MUT and WT samples
    counts_18 = pd.read_csv(path_18m, sep='\t', low_memory=False).set_index('gene_id')
    gene_names = counts_18['gene_name'].copy()
    samples_18 = [c for c in counts_18.columns if c.startswith('M18_MUT') or c.startswith('M18_WT')]
    counts_18 = counts_18[samples_18].apply(pd.to_numeric, errors='coerce').fillna(0).round().astype(int)

    # 3-month: keep MUT and WT, drop the heterozygotes (HET)
    counts_3 = pd.read_csv(path_3m, index_col=0)
    samples_3 = [c for c in counts_3.columns if 'MUT' in c or 'WT' in c]
    counts_3 = counts_3[samples_3].apply(pd.to_numeric, errors='coerce').fillna(0).round().astype(int)

    shared = counts_18.index.intersection(counts_3.index)
    combined = pd.concat([counts_18.loc[shared], counts_3.loc[shared]], axis=1)
    return combined, gene_names

def build_metadata(sample_names):
    """Genotype and age labels per sample, with WT and 3mo as reference levels."""
    meta = pd.DataFrame({'genotype': ['MUT' if 'MUT' in s else 'WT' for s in sample_names], 'age': ['18mo' if s.startswith('M18') else '3mo' for s in sample_names], }, index=sample_names)
    meta['genotype'] = pd.Categorical(meta['genotype'], categories=['WT', 'MUT'])
    meta['age'] = pd.Categorical(meta['age'], categories=['3mo', '18mo'])
    return meta

def contrast_vector(dds, cell_expression):
    """Align a cell-difference expression to the design-matrix columns."""
    if hasattr(cell_expression, 'reindex'):
        cell_expression = cell_expression.reindex(dds.obsm['design_matrix'].columns)
    return np.asarray(cell_expression, dtype=float)

def run_contrast(dds, vector):
    """Run one contrast and return its log2 fold change and Storey q-value."""
    stats = DeseqStats(dds, contrast=vector, quiet=True)
    stats.summary()
    result = stats.results_df.copy()
    result['qvalue'], _ = storey_qvalues(result['pvalue'])
    return result[['log2FoldChange', 'qvalue']]

def group_mean_and_sem(log_norm, sample_ids):
    """Mean and standard error of log2 normalised expression over a set of samples."""
    subset = log_norm[list(sample_ids)]
    return subset.mean(axis=1), subset.std(axis=1, ddof=1) / np.sqrt(subset.shape[1])

def classify(table):
    """Add significance flags and category labels to the per-gene table."""

    def below(col):
        return table[col] < QVALUE_CUT
    table['sig_geno_3mo'] = below('geno_3mo_q')
    table['sig_geno_18mo'] = below('geno_18mo_q')
    table['sig_age_WT'] = below('age_WT_q')
    table['sig_age_MUT'] = below('age_MUT_q')
    table['sig_interaction'] = below('interaction_q')

    # both:          genes where both genotype and age effect are significant
    # genotype_only: genes where MUT != WT regardless of age
    # age_only:      genes that change from 3m to 18m in WT, MUT or both but where MUT != WT at neither age
    # neither:       no significant effect on nay axis (stable genes)

    genotype_sig = table['sig_geno_3mo'] | table['sig_geno_18mo']
    age_sig = table['sig_age_WT'] | table['sig_age_MUT']
    table['affected_by'] = np.select([genotype_sig & age_sig, genotype_sig & ~age_sig, ~genotype_sig & age_sig],
                                     ['both', 'genotype_only', 'age_only'], default='neither')

    # age_dependent_onset: genes not different at 3m but different at 18m
    # transient_early:     genes different at 3m but not different at 18m
    # constitutive:        genes different at both ages in the same direction
    # reversal:            genes different at both ages but in the opposite direction
    # no_genotype_effect:  neither geno_3m nor geno_18m is significant

    s3, s18 = table['sig_geno_3mo'], table['sig_geno_18mo']
    same_direction = np.sign(table['geno_3mo_log2FC']) == np.sign(table['geno_18mo_log2FC'])
    table['interaction_pattern'] = np.select([~s3 & s18, s3 & ~s18, s3 & s18 & same_direction, s3 & s18 & ~same_direction],
                                             ['age_dependent_onset', 'transient_early', 'constitutive', 'reversal'], default='no_genotype_effect')

    table['direction_18mo'] = np.where(table['geno_18mo_log2FC'] > 0, 'up_in_MUT', 'down_in_MUT')
    table['direction_age_WT'] = np.where(table['age_WT_log2FC'] > 0, 'up_with_age', 'down_with_age')
    # WT and MUT move opposite ways with age (for example up in MUT, down in WT)
    table['divergent_trajectory'] = (np.sign(table['age_WT_log2FC']) != np.sign(table['age_MUT_log2FC']))
    return table

def plot_lineplots(table, log_norm, meta, path, top=5):
    """Trajectory panels for the strongest interaction genes plus tp53."""
    # Restrict to genes with a significant genotype x age interaction
    significant = table[(table['interaction_q'] < QVALUE_CUT) & (table['interaction_log2FC'].abs() > LOG2FC_CUT)]

    def is_named(name):
        # Filter out uninformative/placeholder gene symbols so panels show interpretable gene names
        name = str(name)
        return not (name.startswith('si_') or name.startswith('si:') or name.startswith('zgc') or name.lower() == 'nan')

    panels = [(gid, row['gene_name']) for gid, row in significant.sort_values('interaction_q').iterrows() if is_named(row['gene_name'])][:top]
    # Always append tp53 as an extra panel of biological interest, regardless of whether it made the top-N significance cut
    tp53 = table[table['gene_name'].astype(str).str.lower() == 'tp53']
    if len(tp53):
        panels += [(tp53.index[0], 'tp53')]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    colours = {'MUT': '#E63946', 'WT': '#457B9D'}
    for ax, (gene_id, gene_name) in zip(axes.flat, panels):
        expression = log_norm.loc[gene_id].astype(float)
        for genotype in ['WT', 'MUT']:
            means, errors = [], []
            for age in ['3mo', '18mo']:
                # Select samples matching this genotype/age combination and compute mean expression and standard error of the mean (SEM) across replicates
                cols = meta[(meta['genotype'] == genotype) & (meta['age'] == age)].index
                values = expression[cols]
                means.append(values.mean())
                errors.append(values.std(ddof=1) / np.sqrt(max(len(cols), 1)))
            # Plot the 3mo -> 18mo trajectory for this genotype; a genuine interaction shows up as non-parallel ("fanning") MUT vs WT lines
            ax.errorbar([0, 1], means, yerr=errors, marker='o', capsize=4, lw=2, color=colours[genotype], label=genotype)
        row = table.loc[gene_id]
        ax.set_title(f"{gene_name}\nint log2FC={row['interaction_log2FC']:+.2f}, q={row['interaction_q']:.1e}", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['3mo', '18mo'])
        ax.set_ylabel('log2 norm. expr.')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle('Genotype x Age interaction (fanning lines mean interaction)', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_category_counts(table, path):
    """Bar charts of the affected-by and interaction-pattern category counts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Two side-by-side bar charts: one for how many genes fall into each "affected_by" category, one for each "interaction_pattern" category
    for ax, column, title in [(axes[0], 'affected_by', 'Genes affected by'), (axes[1], 'interaction_pattern', 'Interaction pattern')]:
        counts = table[column].value_counts()
        ax.bar(range(len(counts)), counts.values, color='#457B9D')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('number of genes')
        ax.set_title(title, fontweight='bold')
        # Annotate each bar with its exact count
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(int(v)), ha='center', va='bottom', fontsize=8)
        ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()

def plot_genotype_shift(table, path):
    """Genotype effect at 3mo versus 18mo, coloured by interaction pattern."""
    # Fixed colour mapping so interaction-pattern categories are visually consistent across figures; grey is reserved for genes with no genotype effect
    palette = {'age_dependent_onset': '#E63946', 'constitutive': '#2A9D8F', 'reversal': '#8338EC', 'transient_early': '#F4A261', 'no_genotype_effect': '#CCCCCC'}
    fig, ax = plt.subplots(figsize=(7, 6.5))
    # Scatter genes by their 3mo vs 18mo genotype effect, one layer per pattern category, so patterns like "reversal" (sign flip) or "age_dependent_onset" are visible as clusters
    for pattern, colour in palette.items():
        sub = table[table['interaction_pattern'] == pattern]
        ax.scatter(sub['geno_3mo_log2FC'], sub['geno_18mo_log2FC'], s=6, alpha=0.5, color=colour, edgecolors='none', label=f"{pattern} (n={len(sub)})")
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('genotype effect at 3 months (log2FC, MUT vs WT)')
    ax.set_ylabel('genotype effect at 18 months (log2FC, MUT vs WT)')
    ax.set_title('How the genotype effect shifts from 3mo to 18mo', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, markerscale=2, loc='upper left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()

# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Genotype x Age interaction model with per-gene classification")
    print()

    # Load and merge the 18-month (xls) and 3-month (csv) count matrices onto a shared gene set
    combined, gene_names = load_counts(INPUT_18M_XLS, INPUT_3M_CSV)
    meta = build_metadata(combined.columns)
    print("Combined %d samples x %d shared genes" % (combined.shape[1], combined.shape[0]))
    print("Design cells:")
    print(meta.value_counts().to_string())
    print()

    # Filter out low-count genes: keep only genes with at least MIN_COUNT reads in at least MIN_SAMPLES samples, before fitting the model
    keep = (combined >= MIN_COUNT).sum(axis=1) >= MIN_SAMPLES
    counts_filtered = combined.loc[keep]
    print("Genes after filter:", counts_filtered.shape[0])
    print()

    # Fit a single DESeq2 model with a full genotype x age design, so all five contrasts below come from one model rather than five separate fits
    dds = DeseqDataSet(counts=counts_filtered.T, metadata=meta, design="~genotype + age + genotype:age", quiet=True)
    dds.deseq2()

    # Five contrasts from the one fitted model
    interaction = contrast_vector(dds, (dds.cond(genotype='MUT', age='18mo') - dds.cond(genotype='WT', age='18mo')) - (dds.cond(genotype='MUT', age='3mo') - dds.cond(genotype='WT', age='3mo')))
    # Simple genotype effect within each age group
    geno_3mo = contrast_vector(dds, dds.cond(genotype='MUT', age='3mo') - dds.cond(genotype='WT', age='3mo'))
    geno_18mo = contrast_vector(dds, dds.cond(genotype='MUT', age='18mo') - dds.cond(genotype='WT', age='18mo'))
    # Simple age effect within each genotype
    age_wt = contrast_vector(dds, dds.cond(genotype='WT', age='18mo') - dds.cond(genotype='WT', age='3mo'))
    age_mut = contrast_vector(dds, dds.cond(genotype='MUT', age='18mo') - dds.cond(genotype='MUT', age='3mo'))

    # Run Wald tests for each contrast to get log2FC / p-values / q-values per gene
    res_interaction = run_contrast(dds, interaction)
    res_geno_3mo = run_contrast(dds, geno_3mo)
    res_geno_18mo = run_contrast(dds, geno_18mo)
    res_age_wt = run_contrast(dds, age_wt)
    res_age_mut = run_contrast(dds, age_mut)

    # Four group means and SEMs (standard error of the means) from log2 normalised expression
    # Used for plotting trajectories, independent of the DESeq2 contrast statistics above
    normalised = pd.DataFrame(dds.layers['normed_counts'].T, index=counts_filtered.index, columns=counts_filtered.columns)
    log_norm = np.log2(normalised + 1)
    cells = {name: meta[(meta['genotype'] == gt) & (meta['age'] == ag)].index
             for name, (gt, ag) in {'WT_3mo': ('WT', '3mo'), 'MUT_3mo': ('MUT', '3mo'), 'WT_18mo': ('WT', '18mo'), 'MUT_18mo': ('MUT', '18mo')}.items()}

    # Prepare one master table: gene name, per-group mean/SEM expression, and log2FC/qvalue for all five contrasts, indexed by gene ID
    table = pd.DataFrame(index=counts_filtered.index)
    table['gene_name'] = gene_names.reindex(counts_filtered.index).values
    for name, ids in cells.items():
        mean, sem = group_mean_and_sem(log_norm, ids)
        table['mean_' + name] = mean.round(3)
        table['sem_' + name] = sem.round(3)
    table['geno_3mo_log2FC'] = res_geno_3mo['log2FoldChange'].round(4)
    table['geno_3mo_q'] = res_geno_3mo['qvalue']
    table['geno_18mo_log2FC'] = res_geno_18mo['log2FoldChange'].round(4)
    table['geno_18mo_q'] = res_geno_18mo['qvalue']
    table['age_WT_log2FC'] = res_age_wt['log2FoldChange'].round(4)
    table['age_WT_q'] = res_age_wt['qvalue']
    table['age_MUT_log2FC'] = res_age_mut['log2FoldChange'].round(4)
    table['age_MUT_q'] = res_age_mut['qvalue']
    table['interaction_log2FC'] = res_interaction['log2FoldChange'].round(4)
    table['interaction_q'] = res_interaction['qvalue']

    table = classify(table)

    # Interaction-only results, kept for continuity with the earlier output
    interaction_results = table[['interaction_log2FC', 'interaction_q', 'gene_name']].copy()
    interaction_results = interaction_results.rename(columns={'interaction_q': 'qvalue'})
    interaction_results.sort_values('qvalue').to_csv(os.path.join(OUTPUT_DIR, 'interaction_genotype_x_age_results.csv'))
    table.sort_values('interaction_q').to_csv(os.path.join(OUTPUT_DIR, 'interaction_gene_classification.csv'))

    # Category summary: counts of genes in each "affected_by" and "interaction_pattern" category, saved as a single tidy CSV for easy reporting/plotting elsewhere
    summary_rows = []
    for kind, column in [('affected_by', 'affected_by'), ('interaction_pattern', 'interaction_pattern')]:
        for category, count in table[column].value_counts().items():
            summary_rows.append({'kind': kind, 'category': category, 'count': int(count)})
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, 'interaction_category_summary.csv'), index=False)

    # Console summary
    n_interaction = int(table['sig_interaction'].sum())
    print("Significant interaction genes (q<%.2f): %d" % (QVALUE_CUT, n_interaction))
    print()
    print("Affected by:")
    print(table['affected_by'].value_counts().to_string())
    print()
    print("Interaction pattern (how the genotype effect changes with age):")
    print(table['interaction_pattern'].value_counts().to_string())
    print()
    up_mut_down_wt = table[(table['age_MUT_log2FC'] > 0) & (table['age_WT_log2FC'] < 0) & table['sig_interaction']]
    print("Genes up with age in MUT but down with age in WT (sig interaction): %d" % len(up_mut_down_wt))
    print()

    plot_lineplots(table, log_norm, meta, os.path.join(OUTPUT_DIR, 'interaction_lineplots.png'))
    plot_category_counts(table, os.path.join(OUTPUT_DIR, 'interaction_pattern_counts.png'))
    plot_genotype_shift(table, os.path.join(OUTPUT_DIR, 'interaction_geno_shift_scatter.png'))
    print("Saved: interaction_genotype_x_age_results.csv, interaction_gene_classification.csv, interaction_category_summary.csv, and three figures.")
    print()
    print("Outputs in:", OUTPUT_DIR)

if __name__ == '__main__':
    main()
