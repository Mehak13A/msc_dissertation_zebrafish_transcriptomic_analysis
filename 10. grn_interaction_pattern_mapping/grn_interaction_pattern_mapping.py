#!/usr/bin/env python3
"""
10. Mapping the GRN onto the Interaction Model Patterns

PURPOSE
    Connects the two analyses. The GRN model found the transcription factors (TFs) whose targets are dysregulated in
    tert-/- mutants at 18 months. The interaction model classified every gene by how its genotype effect changes with age
    (age-dependent onset, constitutive, and so on). The program asks whether those two pictures describe the same thing:
    are the genes that the GRN's TFs regulate specifically the genes that turn on their genotype effect with age?

    It maps each GRN target (a human symbol) back to its zebrafish orthologue, reads that gene's interaction pattern, and
    compares the pattern distribution of the GRN targets against the genome-wide background (with a Fisher exact test for
    the age-dependent-onset pattern). It also summarises, per TF, how its own targets behave.

INPUTS
    INTERACTION_CSV  : interaction_gene_classification.csv (output of interaction_genotype_x_age.py), the per-gene table
                       with the interaction_pattern, direction_18mo and significance columns.
    ORTHOLOGY_MAP_CSV: orthology_map_zfish_to_human.csv (output of grn_tf_activity_model.py), columns ['zfish_id','human_symbol'].
    GRN_EDGES_CSV    : grn_model_edges.csv (output of grn_tf_activity_model.py), columns ['source','target','mor','target_stat'].
    TF_ACTIVITY_CSV  : tf_activity_Q1.csv (output of grn_tf_activity_model.py), indexed by TF with an 'activity' column.

OUTPUTS (all written to OUTPUT_DIR)
    grn_target_interaction_map.csv- one row per GRN target gene (zebrafish), with its human symbol, how many GRN TFs regulate it, and its interaction pattern, direction and flags.
    grn_tf_pattern_summary.csv    - one row per GRN TF: number of mapped targets, the fraction that are age-dependent onset,
                                    down in MUT and significant interaction, the TF activity, and the TF's own interaction pattern.
    grn_pattern_enrichment.png    - bar chart of the interaction-pattern and affected-by distributions, GRN targets versus the genome background.

METHOD
    Step 1. The GRN is expressed in human symbols and the interaction table in zebrafish gene ids, so they are joined
    through the orthology map: each zebrafish gene is merged with its human orthologue(s). Because of the zebrafish
    whole-genome duplication this is a many-to-many mapping, and all pairs are kept.

    Step 2. A zebrafish gene is called a GRN target if any of its human orthologues is one of the GRN target symbols.
    The gene universe (the background) is every interaction gene that has at least one human orthologue, because only mappable genes could ever be counted as GRN targets.

    Step 3. The interaction pattern of the GRN targets is compared against the background with a Fisher exact test (described below).
    The same genes are also summarised per TF (how that TF's targets behave) and per target gene (its full interaction behaviour).

    FISHER EXACT TEST: The test asks whether GRN targets are more likely than other genes to carry the age-dependent-onset pattern.
    It runs on a two by two count table:
                onset      not onset
    GRN target    a            b
    other genes   c            d

    Fisher exact computes, from the hypergeometric distribution, the probability of seeing a table at least this lopsided
    if target status and onset status were independent. It is exact (no large-sample approximation), so it is valid even
    for small counts, which is why it is the standard choice for gene-set enrichment. The one-sided form tests specifically
    for enrichment. The odds ratio ((a*d)/(b*c)) is the effect size: how many times higher the odds of being onset are among
    targets than among other genes. A small p-value means the enrichment is very unlikely to be a coincidence.

References (this file only)
    These references apply to this source file only and are independent of any reference numbering used in the accompanying report.

    [1] S. Müller-Dott et al., "Expanding the coverage of regulons from high-confidence prior knowledge for accurate estimation
        of transcription factor activities", Nucleic Acids Research, vol. 51, no. 20, pp. 10934-10949, 2023.
    [2] P. Badia-i-Mompel et al., "decoupleR: ensemble of computational methods to infer biological activities from omics data",
        Bioinformatics Advances, vol. 2, no. 1, art. vbac016, 2022.
    [3] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python", Nature Methods, vol. 17, pp. 261-272, 2020.
"""

# Import all necessary libraries and modules

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # non-interactive backend: render figures to files
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

# CONFIGURATION (Input and output paths as per my computer)
INTERACTION_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/9. interaction_genotype_x_age/interaction_gene_classification.csv'
ORTHOLOGY_MAP_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model/orthology_map_zfish_to_human.csv'
GRN_EDGES_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model/grn_model_edges.csv'
TF_ACTIVITY_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model/tf_activity_Q1.csv'
OUTPUT_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/10. grn_interaction_pattern_mapping'

ONSET = 'age_dependent_onset' # Flag genes whose genotype effect only emerges at later ages

def load_orthology_pairs(path):
    """Unique zebrafish-id to human-symbol pairs (upper-case symbols)."""
    ortho = pd.read_csv(path).dropna(subset=['human_symbol'])
    ortho = ortho[~ortho['human_symbol'].isin(['N/A', 'nan', 'None', ''])].copy()
    ortho['human_symbol'] = ortho['human_symbol'].astype(str).str.upper()
    return ortho[['zfish_id', 'human_symbol']].drop_duplicates()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Mapping the GRN onto the interaction model patterns")
    print()

    interaction = pd.read_csv(INTERACTION_CSV)
    interaction = interaction.rename(columns={interaction.columns[0]: 'gene_id'})
    ortho = load_orthology_pairs(ORTHOLOGY_MAP_CSV)
    edges = pd.read_csv(GRN_EDGES_CSV)
    edges['source'] = edges['source'].astype(str).str.upper()
    edges['target'] = edges['target'].astype(str).str.upper()
    tf_activity = pd.read_csv(TF_ACTIVITY_CSV, index_col=0)

    grn_targets = set(edges['target'])
    grn_tfs = set(edges['source'])

    # Attach human symbols to the zebrafish interaction genes (paralogues kept)
    paired = interaction.merge(ortho, left_on='gene_id', right_on='zfish_id', how='inner')
    print("Interaction genes with a human orthologue: %d (from %d)" % (paired['gene_id'].nunique(), len(interaction)))

    # Per zebrafish gene: is it a GRN target, and how many GRN TFs regulate it
    targets_per_gene = (paired[paired['human_symbol'].isin(grn_targets)].merge(edges.groupby('target').size().rename('n_regulating_tfs'), left_on='human_symbol', right_index=True, how='left'))
    is_target = paired.groupby('gene_id')['human_symbol'].apply(lambda s: len(set(s) & grn_targets) > 0)
    per_gene = interaction.set_index('gene_id').copy()
    per_gene['is_grn_target'] = is_target.reindex(per_gene.index).fillna(False)
    mapped_genes = per_gene[per_gene.index.isin(paired['gene_id'])]

    # Enrichment test: are GRN targets more likely than other genes to be age-dependent onset?
    # Build the two by two count table, then run a one-sided Fisher exact test (exact, so valid for any cell counts).
    target_genes = mapped_genes[mapped_genes['is_grn_target']]
    background = mapped_genes # every orthologue-mapped gene
    a = int((target_genes['interaction_pattern'] == ONSET).sum()) # target and onset
    b = int((target_genes['interaction_pattern'] != ONSET).sum()) # target, not onset
    c = int((background['interaction_pattern'] == ONSET).sum()) - a # other genes, onset
    d = int((background['interaction_pattern'] != ONSET).sum()) - b # other genes, not onset
    # odds ratio (a*d)/(b*c) is the effect size; pval is the chance of this enrichment under independence
    odds, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
    frac_target = a / max(a + b, 1)
    frac_bg = (a + c) / max(a + b + c + d, 1)
    print("GRN target genes: %d  (of %d orthologue-mapped genes)" % (a + b, len(mapped_genes)))
    print("Age-dependent onset: %.1f%% of GRN targets vs %.1f%% background" % (100 * frac_target, 100 * frac_bg))
    print("Fisher exact (targets enriched for onset): odds ratio = %.2f, p = %.2e" % (odds, pval))
    print("GRN targets with significant interaction: %.1f%% vs %.1f%% background" % (100 * target_genes['sig_interaction'].mean(), 100 * background['sig_interaction'].mean()))
    print()
    print("Interaction pattern of GRN targets:")
    print(target_genes['interaction_pattern'].value_counts().to_string())
    print()
    print("Direction at 18mo of GRN targets:")
    print(target_genes['direction_18mo'].value_counts().to_string())
    print()

    # Per-target mapping table (zebrafish gene level)
    target_table = targets_per_gene[['gene_id', 'gene_name', 'human_symbol', 'n_regulating_tfs', 'interaction_pattern', 'direction_18mo', 'affected_by', 'geno_18mo_log2FC',
                                     'geno_18mo_q', 'interaction_log2FC', 'interaction_q', 'sig_interaction']].drop_duplicates('gene_id')
    target_table.sort_values('interaction_q').to_csv(os.path.join(OUTPUT_DIR, 'grn_target_interaction_map.csv'), index=False)

    # Per-TF summary
    # Build a lookup from human gene symbol -> that gene's row in the interaction table to check whether a TF itself shows a genotype x age interaction pattern
    pattern_lookup = paired.dropna(subset=['human_symbol']).drop_duplicates('human_symbol').set_index('human_symbol')
    rows = []
    for tf in sorted(grn_tfs):
        # Symbols of all genes this TF regulates, according to the GRN edge list
        tf_target_syms = set(edges[edges['source'] == tf]['target'])
        sub = target_genes[target_genes['human_symbol'].isin(tf_target_syms)] if 'human_symbol' in target_genes else None
        # Map zebrafish target genes to human symbols, then keep only targets of this TF
        sub = mapped_genes.merge(paired[['gene_id', 'human_symbol']], left_index=True, right_on='gene_id')
        sub = sub[sub['human_symbol'].isin(tf_target_syms)]
        n = len(sub)
        # If the TF itself was mapped to an interaction-pattern label, record it; otherwise flag it as unmapped (no human-symbol match)
        tf_pattern = pattern_lookup.loc[tf, 'interaction_pattern'] if tf in pattern_lookup.index else 'unmapped'
        rows.append({
            'TF': tf,
            'tf_activity': round(float(tf_activity['activity'].get(tf, np.nan)), 3) if tf in tf_activity.index else np.nan,
            'n_targets_mapped': n,
            'pct_age_dependent_onset': round(100 * (sub['interaction_pattern'] == ONSET).mean(), 1) if n else 0.0,
            'pct_down_in_MUT': round(100 * (sub['direction_18mo'] == 'down_in_MUT').mean(), 1) if n else 0.0,
            'pct_sig_interaction': round(100 * sub['sig_interaction'].mean(), 1) if n else 0.0,
            'tf_own_interaction_pattern': tf_pattern,
        })
    # Sort and rank TFs by how many of their targets could be mapped, and save the per-TF summary
    pd.DataFrame(rows).sort_values('n_targets_mapped', ascending=False).to_csv(os.path.join(OUTPUT_DIR, 'grn_tf_pattern_summary.csv'), index=False)

    # Figure: GRN targets versus background
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, column, title in [(axes[0], 'interaction_pattern', 'Interaction pattern'), (axes[1], 'affected_by', 'Affected by')]:
        order = background[column].value_counts().index
        tgt = target_genes[column].value_counts(normalize=True).reindex(order).fillna(0) * 100
        bg = background[column].value_counts(normalize=True).reindex(order).fillna(0) * 100
        x = np.arange(len(order))
        ax.bar(x - 0.2, tgt.values, width=0.4, label='GRN targets', color='#E63946')
        ax.bar(x + 0.2, bg.values, width=0.4, label='genome background', color='#BBBBBB')
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('percent of genes')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.2)
    fig.suptitle('GRN target genes are enriched for the age-dependent programme', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, 'grn_pattern_enrichment.png'), dpi=160, bbox_inches='tight')
    plt.close()

    print("Saved: grn_target_interaction_map.csv, grn_tf_pattern_summary.csv, grn_pattern_enrichment.png")
    print()
    print("Outputs in:", OUTPUT_DIR)

if __name__ == '__main__':
    main()