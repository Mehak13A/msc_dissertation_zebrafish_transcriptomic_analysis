#!/usr/bin/env python3
"""
7. STRING Input Preparation and GRN Cross-Reference
Prepares the gene lists that were submitted to the STRING web server and, once STRING's enrichment tables are downloaded,
cross-references them against the gene regulatory network (GRN) model to quantify the overlap between the two
independent methods.

PURPOSE
    The STRING functional-enrichment analysis itself runs on the STRING website (string-db.org), so it is not scripted
    here. What is scripted are the two Python steps around it, so the analysis is reproducible end-to-end apart from the
    website click-through:

    Stage 1 (before STRING): turn the Q1 DEGs into the exact gene lists pasted into STRING: filter to significant DEGs,
                             split into up-/down-in-MUT, and map zebrafish genes to human orthologues.
    Stage 2 (after STRING):  read STRING's enrichment tables back in and count how many genes in the key enriched terms
                             are also targets of the GRN's TFs (the cross-validation reported in the write-up).

    WHY HUMAN SYMBOLS: human STRING is far better annotated than zebrafish, and the GRN was also built on human
                       orthologues, so mapping first keeps STRING and the GRN on the same gene identifiers.

INPUTS
Stage 1:
    Q1_DEG_CSV       - the 18-month Q1 genotype DE table (q1_genotype_DEG_results.csv, output of differential_expression_analysis.py),
                       indexed by zebrafish gene ID, with 'qvalue' and 'log2FoldChange' columns.
    ORTHOLOGY_MAP_CSV- the cached zebrafish -> human map (orthology_map_zfish_to_human.csv, output of grn_tf_activity_model.py), with columns ['zfish_id', 'human_symbol', ...].

Stage 2 (only runs once these exist):
    GRN_EDGES_CSV   - grn_model_edges.csv (output of grn_tf_activity_model.py): columns ['source', 'target', 'mor', 'target_stat'].
    TF_ACTIVITY_CSV - tf_activity_Q1.csv (output of grn_tf_activity_model.py): indexed by TF, with 'activity' and 'pval' columns.
    ENRICHMENT_*_TSV- the 'all enriched terms (without PubMed)' tables exported from STRING for the pooled / up / down runs (10-column TSV).

OUTPUTS (all written to OUTPUT_DIR)
    string_input_human_symbols.txt        : pooled DEG list (to paste into STRING).
    string_input_UP_in_MUT_human.txt      : up-in-MUT list (to paste into STRING).
    string_input_DOWN_in_MUT_human.txt    : down-in-MUT list (to paste into STRING).
    grn_string_overlap_summary.csv        : overlap counts between the key STRING terms and the GRN TF-target sets

HOW TO RUN
    1. Set the Stage-1 paths, then: python string_prep_and_grn_overlap.py writes the three .txt lists.
    2. Paste each list into STRING (string-db.org, "Multiple proteins", organism = Homo sapiens); download each run's "all enriched terms (without PubMed)" TSV.
    3. Set the ENRICHMENT_*_TSV paths (and the GRN paths), then re-run: python string_prep_and_grn_overlap.py -> prints and saves the overlap.

    pip install pandas

References (this file only)
    These references apply to this source file only and are independent of any reference numbering used in the accompanying report.

    [1] D. Szklarczyk et al., "The STRING database in 2023: protein-protein association networks and functional
        enrichment analyses for any sequenced genome of interest", Nucleic Acids Research, vol. 51, no. D1, pp. D638-D646, 2023.
    [2] U. Raudvere et al., "g:Profiler: a web server for functional enrichment analysis and conversions of gene lists
        (2019 update)", Nucleic Acids Research, vol. 47, no. W1, pp. W191-W198, 2019.
"""

# Import all necessary libraries and modules

import os
import pandas as pd

# CONFIGURATION (Input and output paths as per my computer)

# Stage 1 inputs

Q1_DEG_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/1. differential_expression_analysis/q1_genotype_DEG_results.csv'
ORTHOLOGY_MAP_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model/orthology_map_zfish_to_human.csv'

# Stage 2 inputs (GRN artifacts + the TSVs downloaded from STRING; update accordingly when available)
GRN_EDGES_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model/grn_model_edges.csv'
TF_ACTIVITY_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model/tf_activity_Q1.csv'
ENRICHMENT_POOLED_TSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/7. string_prep_and_grn_overlap/Outputs from STRING/enrichment_all.tsv'
ENRICHMENT_UP_TSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/7. string_prep_and_grn_overlap/Outputs from STRING/enrichment_all_up.tsv'
ENRICHMENT_DOWN_TSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/7. string_prep_and_grn_overlap/Outputs from STRING/enrichment_all_down.tsv'

OUTPUT_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/7. string_prep_and_grn_overlap'

# Significance thresholds (match the main DE pipeline and the GRN)
QVALUE_COL = 'qvalue'
LOG2FC_COL = 'log2FoldChange'
QVALUE_CUT = 0.05
LOG2FC_CUT = 1.0
SIG_TF_PVAL = 0.05 # a GRN TF is "significant" if its activity p-value < this

# Column layout of STRING's v12 "all enriched terms" TSV export (10 columns)
ENRICHMENT_COLUMNS = ['category', 'term_id', 'description', 'observed_count', 'background_count', 'strength', 'signal', 'fdr', 'matching_ids', 'matching_labels']

# Stage 1: build STRING inputs

def load_significant_degs(path, qcut, lfccut):
    """Return (all_significant, up_in_mut, down_in_mut) DEG tables."""
    deg = pd.read_csv(path, index_col=0)
    significant = deg[(deg[QVALUE_COL] < qcut) & (deg[LOG2FC_COL].abs() > lfccut)]
    up_in_mut   = significant[significant[LOG2FC_COL] > 0]
    down_in_mut = significant[significant[LOG2FC_COL] < 0]
    return significant, up_in_mut, down_in_mut

def load_orthology_lookup(path):
    """zebrafish gene id -> sorted list of unique human symbols (upper-case)."""
    ortho = pd.read_csv(path).dropna(subset=['human_symbol'])
    ortho = ortho[~ortho['human_symbol'].isin(['N/A', 'nan', 'None', ''])]
    return ortho.groupby('zfish_id')['human_symbol'].apply(lambda symbols: sorted(set(symbols.astype(str).str.upper())))

def degs_to_human_symbols(gene_ids, zfish_to_human):
    """Collapse a set of zebrafish gene ids onto unique human symbols. Returns (sorted_symbols, n_zfish_genes_mapped). 
       A gene can map to several human symbols (the whole-genome duplication), so symbols >= mapped genes."""
    
    symbols = set()
    n_mapped = 0
    for gene_id in gene_ids:
        if gene_id in zfish_to_human.index:
            n_mapped += 1
            symbols.update(zfish_to_human[gene_id])
    return sorted(symbols), n_mapped

def write_symbol_list(symbols, path):
    """Write one gene symbol per line (this is the format STRING's paste box expects)."""
    with open(path, 'w') as handle:
        handle.write("\n".join(symbols))

# Stage 2: GRN cross-reference

def load_enrichment(path):
    """Load a STRING 'all enriched terms' TSV and apply canonical column names."""
    table = pd.read_csv(path, sep='\t')
    if table.shape[1] != len(ENRICHMENT_COLUMNS):
        raise ValueError(f"{os.path.basename(path)} has {table.shape[1]} columns; " f"expected {len(ENRICHMENT_COLUMNS)} (STRING v12 export).")
    table.columns = ENRICHMENT_COLUMNS
    return table

def build_grn_gene_sets(edges_path, tf_activity_path, sig_tf_pval):
    """Build the GRN target sets used for the overlap check."""
    edges = pd.read_csv(edges_path)
    tf_activity = pd.read_csv(tf_activity_path, index_col=0)
    significant_tfs = tf_activity[tf_activity['pval'] < sig_tf_pval]
    activated_tfs = set(significant_tfs[significant_tfs['activity'] > 0].index) # up programme
    repressed_tfs = set(significant_tfs[significant_tfs['activity'] < 0].index) # down programme
    return {
        'E2F_targets': set(edges[edges['source'].astype(str).str.startswith('E2F')]['target']),
        'TP53_targets': set(edges[edges['source'] == 'TP53']['target']),
        'activated_TF_targets': set(edges[edges['source'].isin(activated_tfs)]['target']),
        'repressed_TF_targets': set(edges[edges['source'].isin(repressed_tfs)]['target']),
    }

def term_member_genes(enrichment, category, description):
    """Return the set of the user's genes that belong to one enriched term."""
    row = enrichment[(enrichment['category'] == category) & (enrichment['description'] == description)]
    if row.empty:
        return set()
    return set(str(row.iloc[0]['matching_labels']).split(','))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("STRING input preparation + GRN cross-reference")

    # Stage 1: build the three STRING input lists
    
    significant, up_in_mut, down_in_mut = load_significant_degs(Q1_DEG_CSV, QVALUE_CUT, LOG2FC_CUT)
    print(f"Significant DEGs (q<{QVALUE_CUT}, |log2FC|>{LOG2FC_CUT}): " f"{len(significant)}  ({len(down_in_mut)} down, {len(up_in_mut)} up in MUT)")
    zfish_to_human = load_orthology_lookup(ORTHOLOGY_MAP_CSV)

    for label, subset, filename in [
        ('all',  significant, 'string_input_human_symbols.txt'),
        ('up',   up_in_mut,   'string_input_UP_in_MUT_human.txt'),
        ('down', down_in_mut, 'string_input_DOWN_in_MUT_human.txt'),
    ]:
        symbols, n_mapped = degs_to_human_symbols(subset.index, zfish_to_human)
        write_symbol_list(symbols, os.path.join(OUTPUT_DIR, filename))
        print(f"  {label:4s}: {len(subset):4d} DEGs -> {n_mapped:4d} mapped " f"-> {len(symbols):4d} human symbols  ({filename})")

    # Stage 2: cross-reference STRING enrichment against the GRN
    
    stage2_inputs = [ENRICHMENT_DOWN_TSV, ENRICHMENT_UP_TSV, GRN_EDGES_CSV, TF_ACTIVITY_CSV]
    if not all(os.path.exists(p) for p in stage2_inputs):
        print("\nSTRING enrichment TSVs (or GRN files) not found yet. Paste the three lists into STRING, download each run's 'all enriched terms' TSV, "
              "set the ENRICHMENT_*_TSV paths above, and re-run to compute the GRN overlap.")
        print("\nStage 1 completed! Outputs in:", OUTPUT_DIR)
        return

    grn = build_grn_gene_sets(GRN_EDGES_CSV, TF_ACTIVITY_CSV, SIG_TF_PVAL)
    down_enrichment = load_enrichment(ENRICHMENT_DOWN_TSV)
    up_enrichment = load_enrichment(ENRICHMENT_UP_TSV)

    # Down programme: STRING's cell-cycle term vs the GRN's repressed / E2F targets
    cell_cycle_genes = term_member_genes(down_enrichment, 'Reactome', 'Cell Cycle')
    # Up programme: STRING's stress term vs the GRN's activated / TP53 targets
    response_genes = term_member_genes(up_enrichment, 'GO Process', 'Response to stimulus')

    print("\nGRN cross-reference")
    print(f"DOWN  'Cell Cycle' (Reactome, {len(cell_cycle_genes)} genes): "
          f"{len(cell_cycle_genes & grn['repressed_TF_targets'])} repressed-TF targets, "
          f"{len(cell_cycle_genes & grn['E2F_targets'])} E2F targets")
    print(f"UP    'Response to stimulus' (GO Process, {len(response_genes)} genes): "
          f"{len(response_genes & grn['activated_TF_targets'])} activated-TF targets, "
          f"{len(response_genes & grn['TP53_targets'])} TP53 targets")

    # Save a tidy summary (uniform columns across both directions)
    summary = pd.DataFrame([
        {'direction': 'down', 'string_term': 'Cell Cycle (Reactome)',
         'term_gene_count': len(cell_cycle_genes),
         'primary_grn_set': 'repressed_TF_targets',
         'primary_overlap': len(cell_cycle_genes & grn['repressed_TF_targets']),
         'secondary_grn_set': 'E2F_targets',
         'secondary_overlap': len(cell_cycle_genes & grn['E2F_targets'])},
        {'direction': 'up', 'string_term': 'Response to stimulus (GO Process)',
         'term_gene_count': len(response_genes),
         'primary_grn_set': 'activated_TF_targets',
         'primary_overlap': len(response_genes & grn['activated_TF_targets']),
         'secondary_grn_set': 'TP53_targets',
         'secondary_overlap': len(response_genes & grn['TP53_targets'])},
    ])
    summary.to_csv(os.path.join(OUTPUT_DIR, 'grn_string_overlap_summary.csv'), index=False)
    print("  Saved: grn_string_overlap_summary.csv")
    print("\nProcess finished successfully! Outputs in:", OUTPUT_DIR)

if __name__ == '__main__':
    main()