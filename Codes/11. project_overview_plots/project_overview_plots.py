#!/usr/bin/env python3
"""
11. Project Overview Plots

PURPOSE
    Draws a small set of summary figures that give a surface-level view of the whole project: the sample design (how many
    fish per age and genotype), and the main results across the analyses (differential expression per contrast, the number of
    significant transcription factors per contrast, the interaction-model gene classification, and how the GRN targets map
    onto the age-dependent programme). Every number is computed from the source files, so the figures stay in step with the data.

INPUTS
    COUNTS_18M_XLS     : gene_count.xls (18-month adult-muscle raw counts, with M18/M37/M11 sample columns).
    COUNTS_3M_CSV      : tert3m_raw_counts.csv (3-month raw counts, M3 sample columns).
    Q1_DEG_CSV         : q1_genotype_DEG_results.csv (18-month MUT vs WT).
    Q2_DEG_CSV         : q2_age_DEG_results.csv (old vs young wild-type).
    INTERACTION_CSV    : interaction_gene_classification.csv (per-gene interaction table).
    TF_ACTIVITY_Q1_CSV : tf_activity_Q1.csv (mutant TF activities).
    TF_ACTIVITY_AGE_CSV: tf_activity_age.csv (natural-ageing TF activities).
    GRN_EDGES_CSV      : grn_model_edges.csv (GRN edges).
    ORTHOLOGY_MAP_CSV  : orthology_map_zfish_to_human.csv (zebrafish to human map).

OUTPUTS (all written to OUTPUT_DIR)
    project_sample_design.png   : sample sizes per age and genotype, plus a grid of which groups exist.
    project_tf_counts.png       : number of significant transcription factors per contrast (genotype at 3mo, genotype at 18mo, and age), split into activated and repressed.
    project_results_overview.png: four-panel dashboard - DEGs per contrast, the interaction pattern, the interaction classification, and the GRN target enrichment for the age-dependent pattern.

References (this file only)
    These references apply to this source file only and are independent of any reference numbering used in the accompanying report.

    [1] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," Nature Methods, vol. 17, pp. 261-272, 2020.
"""

# Import all necessary libraries and modules

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # non-interactive backend: render figures to files
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

# CONFIGURATION (Input and output paths as per my computer)
COUNTS_18M_XLS = '/Users/mehakagrawal/Desktop/Final_Dissertation/Datasets/adult_muscle_18mo/gene_count.xls'
COUNTS_3M_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/4. tert_3month_de_analysis/tert3m_raw_counts.csv'
Q1_DEG_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/1. differential_expression_analysis/q1_genotype_DEG_results.csv'
Q2_DEG_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/1. differential_expression_analysis/q2_age_DEG_results.csv'
INTERACTION_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/9. interaction_genotype_x_age/interaction_gene_classification.csv'
TF_ACTIVITY_Q1_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model/tf_activity_Q1.csv'
TF_ACTIVITY_AGE_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/8. q2_age_tf_activity_and_comparison/tf_activity_age.csv'
GRN_EDGES_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model/grn_model_edges.csv'
ORTHOLOGY_MAP_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model/orthology_map_zfish_to_human.csv'
OUTPUT_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/11. project_overview_plots'

QVALUE_CUT = 0.05
LOG2FC_CUT = 1.0
# Standard colour palette: blue for WT, red for MUT, grey for HET, reused so genotype colouring stays consistent across all figures
COL_WT, COL_MUT, COL_HET = '#457B9D', '#E63946', '#999999'
COL_UP, COL_DOWN = '#E63946', '#457B9D' # Up/down-regulation colours
AGES = ['3mo', '11mo', '18mo', '37mo']
GENOTYPES = ['WT', 'MUT', 'HET']
ONSET = 'age_dependent_onset' # Flag genes whose genotype effect only emerges at later ages

def count_samples():
    """Number of samples per (age, genotype) across both count matrices."""
    # Initialise a nested dictionary grid[genotype][age] = 0, to be filled by scanning column headers
    grid = {g: {a: 0 for a in AGES} for g in GENOTYPES}
    for path, sep in [(COUNTS_18M_XLS, '\t'), (COUNTS_3M_CSV, ',')]:
        # Only read the header row (nrows=0)
        columns = pd.read_csv(path, sep=sep, nrows=0).columns
        for col in columns:
            # Use of regex to match sample column names like "M18_MUT_1" with age code "18", genotype "MUT"
            match = re.match(r'M(\d+)_([A-Za-z]+)', str(col))
            if not match:
                continue
            age = 'M' + match.group(1)
            age = age.replace('M', '') + 'mo' # e.g. "18" -> "18mo"
            genotype = match.group(2).upper()
            if age in grid.get(genotype, {}):
                grid[genotype][age] += 1
    return grid

def deg_counts(path):
    """Total, up and down significant DEGs for a standard results table."""
    d = pd.read_csv(path, index_col=0)
    # Apply the standard significance filter (q < cutoff, |log2FC| > cutoff)
    sig = d[(d['qvalue'] < QVALUE_CUT) & (d['log2FoldChange'].abs() > LOG2FC_CUT)]
    # Return (total DEGs, upregulated count, downregulated count)
    return len(sig), int((sig['log2FoldChange'] > 0).sum()), int((sig['log2FoldChange'] < 0).sum())

def tf_counts(path):
    """Significant TFs split into activated and repressed."""
    t = pd.read_csv(path, index_col=0)
    # TF significance uses a raw p-value cutoff here (not q-value), per the TF activity output format
    sig = t[t['pval'] < QVALUE_CUT]
    # Return (total TFs tested, significant activated count, significant repressed count)
    return len(t), int((sig['activity'] > 0).sum()), int((sig['activity'] < 0).sum())

def grn_onset_enrichment(interaction, ortho, edges):
    """Fraction of GRN targets that are age-dependent onset, versus background."""
    # Standardise the interaction table's gene ID column name
    interaction = interaction.rename(columns={interaction.columns[0]: 'gene_id'})
    # Clean the zebrafish-to-human orthologue mapping table: drop unmapped genes
    ortho = ortho.dropna(subset=['human_symbol']).copy()
    ortho = ortho[~ortho['human_symbol'].isin(['N/A', 'nan', 'None', ''])] # remove placeholder/missing symbol strings
    ortho['human_symbol'] = ortho['human_symbol'].astype(str).str.upper() # standardise casing for matching
    # Join interaction results to human gene symbols via the ortholog table
    paired = interaction.merge(ortho[['zfish_id', 'human_symbol']].drop_duplicates(), left_on='gene_id', right_on='zfish_id', how='inner')
    grn_targets = set(edges['target'].astype(str).str.upper())
    # For each zebrafish gene (which may map to multiple human symbols/paralogs), flag it as a GRN target if any of its mapped symbols is a known target
    is_target = paired.groupby('gene_id')['human_symbol'].apply(lambda s: len(set(s) & grn_targets) > 0)
    mapped = interaction.set_index('gene_id').loc[is_target.index].copy()
    mapped['is_target'] = is_target
    target = mapped[mapped['is_target']]
    # Build a 2x2 contingency table for Fisher's exact test:
    a = int((target['interaction_pattern'] == ONSET).sum()) # GRN targets with onset pattern
    b = int((target['interaction_pattern'] != ONSET).sum()) # GRN targets without onset pattern
    c = int((mapped['interaction_pattern'] == ONSET).sum()) - a # non-targets with onset pattern
    d = int((mapped['interaction_pattern'] != ONSET).sum()) - b # non-targets without onset pattern

    # One-sided Fisher's exact test: are GRN targets enriched for the onset pattern relative to the background (non-target) genes?
    odds, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
    # Percentage of onset-pattern genes among targets vs among all mapped genes (background)
    frac_target = 100 * a / max(a + b, 1)
    frac_background = 100 * (a + c) / max(a + b + c + d, 1)
    return frac_target, frac_background, odds, pval, len(grn_targets)

def figure_sample_design(sample_grid, path):
    """ Two-panel figure: a grouped bar chart of sample counts, plus a coverage grid showing which (genotype, age) combinations actually exist in the dataset"""
    fig, (ax_bar, ax_grid) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: grouped bar chart of sample counts per age, one bar cluster per genotype
    x = np.arange(len(AGES))
    x = np.arange(len(AGES))
    width = 0.26
    for i, (genotype, colour) in enumerate([('WT', COL_WT), ('MUT', COL_MUT), ('HET', COL_HET)]):
        values = [sample_grid[genotype][a] for a in AGES]
        bars = ax_bar.bar(x + (i - 1) * width, values, width, color=colour, label=genotype)
        for bar, value in zip(bars, values):
            if value > 0:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha='center', va='bottom', fontsize=9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(AGES)
    ax_bar.set_ylabel('number of fish (samples)')
    ax_bar.set_title('Sample sizes per age and genotype', fontweight='bold')
    ax_bar.legend()
    ax_bar.grid(True, axis='y', alpha=0.2)
    ax_bar.set_ylim(0, max(max(v.values()) for v in sample_grid.values()) + 1)

    # Right panel: presence/absence grid showing which genotype x age cells have data
    grid = np.array([[sample_grid[g][a] for a in AGES] for g in GENOTYPES], float)
    ax_grid.imshow(np.where(grid > 0, 1, 0), cmap='Greens', vmin=0, vmax=2, aspect='auto')
    for r, genotype in enumerate(GENOTYPES):
        for c_, age in enumerate(AGES):
            value = int(grid[r, c_])
            ax_grid.text(c_, r, ('n=%d' % value) if value > 0 else 'absent', ha='center', va='center', fontsize=10, color='black' if value > 0 else '#888888', fontweight='bold' if value > 0 else 'normal')
    ax_grid.set_xticks(range(len(AGES)))
    ax_grid.set_xticklabels(AGES)
    ax_grid.set_yticks(range(len(GENOTYPES)))
    ax_grid.set_yticklabels(GENOTYPES)
    ax_grid.set_xlabel('age')
    ax_grid.set_ylabel('genotype')
    ax_grid.set_title('Design coverage (which groups exist)', fontweight='bold')
    fig.suptitle('Study design: zebrafish flank muscle RNA-seq', fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()

def figure_tf_counts(tf_q1, tf_age, n_3mo_degs, path):
    """Significant TFs per contrast.
    TF activity is a property of a CONTRAST (one group compared with another), not of a single group, so the counts are shown per comparison.
    The 3-month genotype comparison is included to show that the mutant has essentially no transcriptional signal when young.
    """
    # Unpack (total_scored, n_activated, n_repressed) tuples from tf_counts() for each contrast
    _, act_q1, rep_q1 = tf_q1
    _, act_age, rep_age = tf_age
    labels = ['Genotype effect\n(mutant vs WT, 3mo)', 'Genotype effect\n(mutant vs WT, 18mo)', 'Age effect\n(old vs young WT)']
    activated = [0, act_q1, act_age]
    repressed = [0, rep_q1, rep_age]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 6))
    # Stacked bars: repressed TFs as the base segment, activated TFs stacked on top
    ax.bar(x, repressed, color=COL_WT, label='repressed')
    ax.bar(x, activated, bottom=repressed, color=COL_MUT, label='activated')
    for i in range(len(labels)):
        total = activated[i] + repressed[i]
        if total == 0:
            ax.text(i, 1.5, 'no signal\n(%d DEGs at 3mo)' % n_3mo_degs, ha='center', va='bottom', fontsize=9, style='italic', color='#666666')
        else:
            ax.text(i, total + 1.5, '%d significant TFs' % total, ha='center', fontsize=11, fontweight='bold')
            if repressed[i] > 0:
                ax.text(i, repressed[i] / 2, '%d repressed' % repressed[i], ha='center', va='center', fontsize=9, color='white')
            if activated[i] > 0:
                ax.text(i, repressed[i] + activated[i] / 2, '%d activated' % activated[i], ha='center', va='center', fontsize=9, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('significant transcription factors (p<%.2f, of %d scored)' % (QVALUE_CUT, tf_q1[0]))
    ax.set_title('Transcription factors by contrast', fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.2)
    ax.set_ylim(0, max(sum(tf_q1[1:]), sum(tf_age[1:])) + 14)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()

def figure_results_overview(deg, interaction, pattern, grn, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: DEGs per contrast
    ax = axes[0, 0]
    labels = ['18mo genotype\n(MUT vs WT)', 'age\n(old vs young WT)', '3mo genotype\n(MUT vs WT)', 'interaction\n(genotype x age)']
    up = [deg['q1'][1], deg['q2'][1], deg['g3'][1], deg['int'][1]]
    down = [deg['q1'][2], deg['q2'][2], deg['g3'][2], deg['int'][2]]
    x = np.arange(len(labels))
    ax.bar(x, down, color=COL_DOWN, label='down / negative')
    ax.bar(x, up, bottom=down, color=COL_UP, label='up / positive')
    for i in range(len(labels)):
        ax.text(i, up[i] + down[i] + 40, str(up[i] + down[i]), ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('significant genes (q<0.05, |log2FC|>1)')
    ax.set_title('Differential expression per contrast', fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.2)

    # Panel 2: interaction pattern (how the genotype effect changes with age)
    ax = axes[0, 1]
    pattern_order = ['no_genotype_effect', 'age_dependent_onset', 'transient_early', 'reversal', 'constitutive']
    pattern_colours = ['#CCCCCC', '#E63946', '#F4A261', '#8338EC', '#2A9D8F']
    pattern_values = [pattern.get(p, 0) for p in pattern_order]
    ax.bar(range(len(pattern_order)), pattern_values, color=pattern_colours)
    for i, v in enumerate(pattern_values):
        ax.text(i, v + 150, str(v), ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(pattern_order)))
    ax.set_xticklabels([p.replace('_', '\n') for p in pattern_order], fontsize=8)
    ax.set_ylabel('number of genes')
    ax.set_title('Interaction pattern: how the genotype effect changes with age', fontweight='bold', fontsize=10)
    ax.grid(True, axis='y', alpha=0.2)

    # Panel 3: interaction classification
    ax = axes[1, 0]
    order = ['age_only', 'both', 'neither', 'genotype_only']
    colours = ['#F4A261', '#2A9D8F', '#CCCCCC', '#E63946']
    values = [interaction.get(o, 0) for o in order]
    ax.bar(range(len(order)), values, color=colours)
    for i, v in enumerate(values):
        ax.text(i, v + 80, str(v), ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=15, fontsize=9)
    ax.set_ylabel('number of genes')
    ax.set_title('Interaction model: genes affected by', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2)

    # Panel 4: GRN target enrichment
    ax = axes[1, 1]
    frac_target, frac_bg, odds, pval, n_targets = grn
    ax.bar([0, 1], [frac_target, frac_bg], color=[COL_MUT, '#BBBBBB'], width=0.5)
    ax.text(0, frac_target + 1.2, '%.1f%%' % frac_target, ha='center', fontweight='bold')
    ax.text(1, frac_bg + 1.2, '%.1f%%' % frac_bg, ha='center', fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['GRN targets', 'genome\nbackground'])
    ax.set_ylabel('percent age-dependent onset')
    ax.set_ylim(0, max(frac_target, frac_bg) + 12)
    ax.set_title('GRN targets vs genome: age-dependent onset\n(%d targets; odds ratio=%.1f, p=%.0e)' % (n_targets, odds, pval), fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2)

    fig.suptitle('Results overview across all analyses', fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Building project overview plots")
    print()

    # Sample counts per (genotype, age), scanned from the raw count matrix headers
    sample_grid = count_samples()
    interaction_table = pd.read_csv(INTERACTION_CSV)
    interaction_indexed = interaction_table.set_index(interaction_table.columns[0])

    # DEG counts: Q1 and Q2 from their tables; 3mo and interaction from the classification
    g3_up = int((interaction_indexed['sig_geno_3mo'] & (interaction_indexed['geno_3mo_log2FC'] > 0)).sum())
    g3_down = int((interaction_indexed['sig_geno_3mo'] & (interaction_indexed['geno_3mo_log2FC'] < 0)).sum())
    # Same approach for significant interaction genes
    int_up = int((interaction_indexed['sig_interaction'] & (interaction_indexed['interaction_log2FC'] > 0)).sum())
    int_down = int((interaction_indexed['sig_interaction'] & (interaction_indexed['interaction_log2FC'] < 0)).sum())
    deg = {'q1': deg_counts(Q1_DEG_CSV), 'q2': deg_counts(Q2_DEG_CSV), 'g3': (g3_up + g3_down, g3_up, g3_down), 'int': (int_up + int_down, int_up, int_down)}

    # Significant TF counts for the genotype-effect and age-effect TF activity tables
    tf_q1 = tf_counts(TF_ACTIVITY_Q1_CSV)
    tf_age = tf_counts(TF_ACTIVITY_AGE_CSV)

    # Gene counts by "affected_by" category and "interaction_pattern" category, for the results-overview figure
    affected = interaction_indexed['affected_by'].value_counts().to_dict()
    pattern = interaction_indexed['interaction_pattern'].value_counts().to_dict()
    grn = grn_onset_enrichment(interaction_table.copy(), pd.read_csv(ORTHOLOGY_MAP_CSV), pd.read_csv(GRN_EDGES_CSV))

    figure_sample_design(sample_grid, os.path.join(OUTPUT_DIR, 'project_sample_design.png'))
    figure_tf_counts(tf_q1, tf_age, deg['g3'][0], os.path.join(OUTPUT_DIR, 'project_tf_counts.png'))
    figure_results_overview(deg, affected, pattern, grn, os.path.join(OUTPUT_DIR, 'project_results_overview.png'))

    # Console summary
    print("Sample counts per (genotype, age):")
    for g in GENOTYPES:
        print("  %-4s %s" % (g, {a: sample_grid[g][a] for a in AGES}))
    print()
    print("DEGs: Q1=%d, Q2=%d, 3mo=%d, interaction=%d" % (deg['q1'][0], deg['q2'][0], deg['g3'][0], deg['int'][0]))
    print("Significant TFs: genotype=%d, age=%d (of %d scored)" % (tf_q1[1] + tf_q1[2], tf_age[1] + tf_age[2], tf_q1[0]))
    print("GRN targets age-dependent onset: %.1f%% vs %.1f%% background" % (grn[0], grn[1]))
    print()
    print("Saved: project_sample_design.png, project_tf_counts.png, project_results_overview.png")
    print("Outputs in:", OUTPUT_DIR)

if __name__ == '__main__':
    main()
