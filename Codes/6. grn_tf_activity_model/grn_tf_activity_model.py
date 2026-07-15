#!/usr/bin/env python3
"""
6. GRN-based Transcription Factor Activity Model
Zebrafish to human orthologue mapping + curated human regulons (CollecTRI) + transcription-factor activity inference (decoupler).

PURPOSE
    Turn the Q1 (genotype) differential-expression result into a transcription-factor (TF) activity model. The 18-month
    sample size (4 MUT vs 5 WT) is far too small to infer a gene regulatory network (GRN) from the data itself, so instead
    of learning edges from the data this script adopts a pre-built, curated human regulatory network (CollecTRI [3]) and
    uses the data only to SCORE it: for each TF, are its target genes collectively up- or down-regulated in mutants? This
    yields (a) a ranked TF-activity table for Q1 and (b) a reusable network artifact that can be re-applied to any other dataset.

INPUTS
    Q1_CSV: the 18-month Q1 genotype DE table (output of differential_expression_analysis.py, q1_genotype_DEG_results.csv),
            indexed by zebrafish gene ID, with a gene-level statistic column ('stat', falling back to 'log2FoldChange'),
            a 'qvalue' column and a 'log2FoldChange' column.
    SECOND_DATASET_CSV: a second DE table to demonstrate model reuse, e.g. tert3m_MUT_vs_WT_results.csv (output of tert_3month_de_analysis.py).
    TF_LIST_CS: an existing TF list for a sanity-check overlap, e.g. q1_transcription_factors.csv (output of go_enrichment_analysis.py).

    Requires an internet connection: g:Profiler orthology [1] is an online service and decoupler downloads CollecTRI via OmniPath.

OUTPUTS (all written to OUTPUT_DIR)
    orthology_map_zfish_to_human.csv: cached zebrafish -> human orthologue map (reused on subsequent runs; delete to force a re-map).
    tf_activity_Q1.csv              : ranked TF activities and p-values for Q1.
    tf_activity_barplot_Q1.png      : top activated / repressed TFs.
    grn_model_edges.csv             : the reusable network as an edge list (significant TFs -> dysregulated targets).
    grn_model.graphml               : the same network as GraphML (to be run in Cytoscape [5]).
    tf_activity_dataset2.csv        : TF activities for the second dataset.
    tf_activity_barplot_dataset2.png: top activated / repressed TFs for the second dataset.

METHOD
    Zebrafish gene IDs are mapped to human orthologues with g:Profiler g:Orth [1]; the full gene-level statistic vector
    (all mapped genes, not only DEGs) becomes the input to decoupler's univariate linear model (ulm) [2], scored against
    the CollecTRI regulons [3]. TFs with activity p < SIG_TF_PVAL are called significant. The curated network is then
    subset to those significant TFs and the dysregulated (q < QVALUE_CUT, |log2FC| > LOG2FC_CUT) target genes to give a
    compact, reusable GRN artifact, exported as an edge list and as GraphML via NetworkX [4].

References (this file only)
    These references apply to this source file only and are independent of any reference numbering used in the accompanying report.

    [1] U. Raudvere et al., "g:Profiler: a web server for functional enrichment analysis and conversions of gene lists
        (2019 update)", Nucleic Acids Research, vol. 47, no. W1, pp. W191-W198, 2019.
    [2] P. Badia-i-Mompel et al., "decoupleR: ensemble of computational methods to infer biological activities from 
        omics data", Bioinformatics Advances, vol. 2, no. 1, Art. no. vbac016, 2022, doi: 10.1093/bioadv/vbac016.
    [3] S. Müller-Dott et al., "Expanding the coverage of regulons from high-confidence prior knowledge for accurate
        estimation of transcription factor activities", Nucleic Acids Research, vol. 51, no. 20, pp. 10934-10949, 2023.
    [4] A. A. Hagberg, D. A. Schult, and P. J. Swart, "Exploring network structure, dynamics, and function using
        NetworkX", in Proc. 7th Python in Science Conf. (SciPy 2008), pp. 11-15, 2008.
    [5] P. Shannon et al., "Cytoscape: a software environment for integrated models of biomolecular interaction networks",
        Genome Research, vol. 13, no. 11, pp. 2498–2504, 2003.
"""

# Import all necessary libraries and modules

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # non-interactive backend: render figures to files
import matplotlib.pyplot as plt

# CONFIGURATION (Input and output paths as per my computer)

# Q1 (18-month genotype) DE table - output of differential_expression_analysis.py
Q1_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/1. differential_expression_analysis/q1_genotype_DEG_results.csv'
OUTPUT_DIR = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/6. grn_tf_activity_model'

# Second DE table for reuse, e.g. tert3m_MUT_vs_WT_results.csv (tert_3month_de_analysis.py output)
SECOND_DATASET_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/4. tert_3month_de_analysis/tert3m_MUT_vs_WT_results.csv'
# Existing TF list for the sanity-check overlap, e.g. q1_transcription_factors.csv (go_enrichment_analysis.py output)
TF_LIST_CSV = '/Users/mehakagrawal/Desktop/Final_Dissertation/Outputs/3. go_enrichment_analysis/q1_transcription_factors.csv'

# Analysis settings
STAT_COL = 'stat' # gene-level signal fed to decoupler ('stat' preferred; falls back to 'log2FoldChange')
QVALUE_COL = 'qvalue'
LOG2FC_COL = 'log2FoldChange'
QVALUE_CUT = 0.05
LOG2FC_CUT = 1.0
SIG_TF_PVAL = 0.05 # a TF is "significant" if its activity p-value < this
SOURCE_ORG = 'drerio' # zebrafish
TARGET_ORG = 'hsapiens' # human
AGG = 'maxabs' # how to collapse many zebrafish genes to one human symbol ('maxabs' or 'mean')

# Helper Functions

def _pick_col(columns, candidates):
    """Return the first candidate present in columns, else None."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None

def build_human_stat_vector(deg_df, ortho_map, stat_col, agg='maxabs'):
    """Collapse a zebrafish gene-level statistic onto human symbols.

    deg_df    : DataFrame indexed by zebrafish gene id, containing `stat_col`.
    ortho_map : DataFrame with columns ['zfish_id', 'human_symbol'].
    Returns   : Series indexed by human symbol -> statistic.
    """
    # Gene-level statistic per zebrafish gene, dropping genes with no value.
    stat_series = deg_df[stat_col].dropna()
    # Keeps only rows with a usable human symbol.
    ortho_clean = ortho_map.dropna(subset=['human_symbol']).copy()
    ortho_clean = ortho_clean[~ortho_clean['human_symbol'].isin(['N/A', 'nan', 'None', ''])]
    # Attach each zebrafish gene's statistic to its human symbol.
    merged = ortho_clean.merge(stat_series.rename('stat'), left_on='zfish_id', right_index=True, how='inner')
    if merged.empty:
        return pd.Series(dtype=float)
    if agg == 'maxabs':
        # Many zebrafish genes can map to one human symbol. Keep the strongest signal.
        merged['absstat'] = merged['stat'].abs()
        merged = merged.sort_values('absstat', ascending=False).drop_duplicates('human_symbol', keep='first')
        collapsed = merged.set_index('human_symbol')['stat']
    else:  # mean: average the statistics of all zebrafish genes mapping to a symbol
        collapsed = merged.groupby('human_symbol')['stat'].mean()
    return collapsed.astype(float)

def build_grn_edges(collectri, sig_tfs, target_stat):
    """Subset the curated network to significant TFs and measured targets.

    collectri   : DataFrame with columns ['source', 'target', 'weight'].
    sig_tfs     : iterable of human TF symbols deemed active/repressed.
    target_stat : Series human_symbol -> statistic (the targets we keep).
    Returns     : edge DataFrame ['source','target','mor','target_stat'].
    """
    sig_tfs = set(sig_tfs)
    keep_targets = set(target_stat.index)
    # Keep only edges whose TF is significant AND whose target we measured.
    edges = collectri[collectri['source'].isin(sig_tfs) & collectri['target'].isin(keep_targets)].copy()
    edges = edges.rename(columns={'weight': 'mor'}) # mode of regulation (+1 activating / -1 repressing)
    edges['target_stat'] = edges['target'].map(target_stat) # attach each target's statistic
    return edges[['source', 'target', 'mor', 'target_stat']].reset_index(drop=True)

# Network / IO steps

def map_orthologues(zfish_ids, source_org, target_org, batch=2000):
    """Map zebrafish gene IDs to human symbols via g:Profiler g:Orth (online)."""
    from gprofiler import GProfiler
    profiler = GProfiler(return_dataframe=True)
    unique_ids = list(dict.fromkeys(zfish_ids)) # de-duplicate, preserving order
    batch_frames = []
    # g:Orth is queried in batches to keep each request a manageable size.
    for start in range(0, len(unique_ids), batch):
        id_chunk = unique_ids[start:start + batch]
        batch_result = profiler.orth(organism=source_org, query=id_chunk, target=target_org)
        batch_frames.append(batch_result)
        print(f"  orthology: mapped batch {start // batch + 1} ({start + len(id_chunk)}/{len(unique_ids)})")
    raw_map = pd.concat(batch_frames, ignore_index=True)
    # g:Profiler column names vary by version hence find the ones we need.
    incoming_col = _pick_col(raw_map.columns, ['incoming'])
    symbol_col   = _pick_col(raw_map.columns, ['name', 'ortholog_name'])
    ensg_col     = _pick_col(raw_map.columns, ['converted', 'ortholog_ensg', 'ensg'])
    if incoming_col is None or symbol_col is None:
        raise RuntimeError(f"Unexpected g:Orth columns: {list(raw_map.columns)}")
    ortho = pd.DataFrame({'zfish_id': raw_map[incoming_col], 'human_symbol': raw_map[symbol_col], 'human_ensg': raw_map[ensg_col] if ensg_col else '',})
    # Drop rows with no real orthologue, then de-duplicate.
    ortho = ortho[~ortho['human_symbol'].isin(['N/A', 'nan', 'None', ''])].dropna(subset=['human_symbol'])
    return ortho.drop_duplicates()

def get_collectri_net():
    """Load CollecTRI (human) across decoupler v1.x and v2.x APIs."""
    import decoupler as dc
    if hasattr(dc, 'get_collectri'): # decoupler 1.x
        network = dc.get_collectri(organism='human', split_complexes=False)
    elif hasattr(dc, 'op') and hasattr(dc.op, 'collectri'): # decoupler 2.x
        network = dc.op.collectri(organism='human')
    else:
        raise RuntimeError("This decoupler version exposes no CollecTRI loader (expected dc.get_collectri or dc.op.collectri).")
    # Normalise the weight column name across versions.
    if 'weight' not in network.columns and 'mor' in network.columns:
        network = network.rename(columns={'mor': 'weight'})
    return network

def run_tf_activity(human_stat, collectri, label):
    """Infer TF activities from one gene-level statistic vector. Works with both decoupler 1.x (dc.run_ulm) and 2.x (dc.mt.ulm)."""

    import decoupler as dc
    # decoupler expects a samples x genes matrix; here that is a single sample.
    sample_by_gene = human_stat.rename(label).to_frame().T.astype(float)
    if hasattr(dc, 'run_ulm'): # decoupler 1.x
        activities, pvalues = dc.run_ulm(mat=sample_by_gene, net=collectri, source='source', target='target', weight='weight', min_n=5, verbose=True)
    elif hasattr(dc, 'mt') and hasattr(dc.mt, 'ulm'): # decoupler 2.x
        activities, pvalues = dc.mt.ulm(data=sample_by_gene, net=collectri)
    else:
        raise RuntimeError("This decoupler version exposes no ulm method (expected dc.run_ulm or dc.mt.ulm).")
    # One row per sample -> take our single sample's activities and p-values.
    result = pd.DataFrame({'activity': activities.loc[label], 'pval': pvalues.loc[label]})
    return result.sort_values('activity', ascending=False)

def plot_tf_bar(tf_activity, label, path, top=15):
    """Horizontal bar chart of the most activated / repressed TFs."""

    # Take the strongest positive and negative TFs, de-duplicate, order for plotting.
    top_tfs = pd.concat([tf_activity.head(top), tf_activity.tail(top)])
    top_tfs = top_tfs[~top_tfs.index.duplicated()].sort_values('activity')
    bar_colors = ['#457B9D' if v < 0 else '#E63946' for v in top_tfs['activity']] # blue=repressed, red=activated
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top_tfs.index, top_tfs['activity'], color=bar_colors)
    ax.axvline(0, color='black', lw=0.8)
    # ulm score stands for Univariate Linear Model score. It is a metric to estimate whether a specific TF is active or
    # inactive based on gene expression data. A positive score indicates that the TF and its target genes are upregulated,
    # (TF is likely active) whereas a negative score means the TF's target genes are downregulated (TF is likely inactive).
    ax.set_xlabel('Inferred TF activity (ulm score)')
    ax.set_title(f'Top activated (red) / repressed (blue) TFs - {label}', fontweight='bold')
    ax.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")

def export_graphml(edges, tf_activity, path):
    """Write the GRN model as GraphML (to be run in Cytoscape [5])."""
    import networkx as nx
    graph = nx.DiGraph()
    for _, edge_row in edges.iterrows():
        tf, target = str(edge_row['source']), str(edge_row['target'])
        # TF node carries its inferred activity score.
        tf_act = float(tf_activity['activity'].get(tf, 0.0)) if tf in tf_activity.index else 0.0
        graph.add_node(tf, node_type='TF', tf_activity=tf_act)
        # Target node carries its gene-level statistic (unless it is itself a TF node).
        target_stat_value = float(edge_row['target_stat']) if pd.notna(edge_row['target_stat']) else 0.0
        if not graph.has_node(target) or graph.nodes[target].get('node_type') != 'TF':
            graph.add_node(target, node_type='target', target_stat=target_stat_value)
        graph.add_edge(tf, target, mor=int(edge_row['mor'])) # edge weight = mode of regulation
    nx.write_graphml(graph, path)
    print(f"  Saved: {os.path.basename(path)}  ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")

# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("GRN / TF-activity model  (zebrafish -> human orthologue mapping)")

    # 1. Load DEG results
    deg_results = pd.read_csv(Q1_CSV, index_col=0)
    stat_col = STAT_COL if STAT_COL in deg_results.columns else LOG2FC_COL
    if stat_col != STAT_COL:
        print(f"NOTE: '{STAT_COL}' not found; using '{LOG2FC_COL}' as the gene-level signal.")
    print(f"Loaded {len(deg_results)} genes from Q1.")

    # 2. Orthologue mapping (zebrafish -> human): cached after the first run
    ortho_cache_path = os.path.join(OUTPUT_DIR, 'orthology_map_zfish_to_human.csv')
    if os.path.exists(ortho_cache_path):
        ortho_map = pd.read_csv(ortho_cache_path)
        print(f"\nLoaded cached orthology map ({ortho_map['zfish_id'].nunique()} genes). Delete that CSV to force a re-map.")
    else:
        print("\nMapping zebrafish gene IDs to human orthologues via g:Profiler...")
        ortho_map = map_orthologues(list(deg_results.index), SOURCE_ORG, TARGET_ORG)
        # Attach the zebrafish gene name (if present) for readability of the cached map.
        ortho_map = ortho_map.merge(deg_results[[c for c in ['gene_name'] if c in deg_results.columns]], left_on='zfish_id', right_index=True, how='left')
        ortho_map.to_csv(ortho_cache_path, index=False)
        print("  Saved: orthology_map_zfish_to_human.csv")
    n_mapped = ortho_map['zfish_id'].nunique()
    print(f"  {n_mapped}/{len(deg_results)} zebrafish genes mapped " f"({100 * n_mapped / len(deg_results):.1f}%).")

    # 3. Build the human gene-level statistic vector (all mapped genes -> background for ulm)
    human_stat = build_human_stat_vector(deg_results, ortho_map, stat_col, agg=AGG)
    print(f"\nHuman gene-level vector: {len(human_stat)} unique symbols.")

    # 4. Get curated regulons (CollecTRI, human)
    print("\nDownloading CollecTRI regulons (human) via decoupler/OmniPath...")
    collectri = get_collectri_net()
    print(f"  CollecTRI: {collectri['source'].nunique()} TFs, {len(collectri)} interactions.")

    # 5. Infer TF activities for Q1
    print("\nInferring TF activities (ulm)...")
    tf_activity = run_tf_activity(human_stat, collectri, label='Q1_MUT_vs_WT')
    tf_activity.to_csv(os.path.join(OUTPUT_DIR, 'tf_activity_Q1.csv'))
    print("  Saved: tf_activity_Q1.csv")
    significant_tfs = tf_activity[tf_activity['pval'] < SIG_TF_PVAL]
    print(f"  Significant TFs (p<{SIG_TF_PVAL}): {len(significant_tfs)}")
    print("  Most activated in MUT:", ", ".join(tf_activity.head(8).index))
    print("  Most repressed in MUT:", ", ".join(tf_activity.tail(8).index[::-1]))
    plot_tf_bar(tf_activity, 'Q1_MUT_vs_WT', os.path.join(OUTPUT_DIR, 'tf_activity_barplot_Q1.png'))

    # 6. Build the reusable GRN model artifact (sig TFs -> dysregulated targets)
    print("\nBuilding GRN model artifact...")
    significant_mask = (deg_results[QVALUE_COL] < QVALUE_CUT) & (deg_results[LOG2FC_COL].abs() > LOG2FC_CUT)
    sig_target_stat = build_human_stat_vector(deg_results.loc[significant_mask], ortho_map, stat_col, agg=AGG)
    edges = build_grn_edges(collectri, significant_tfs.index, sig_target_stat)
    edges.to_csv(os.path.join(OUTPUT_DIR, 'grn_model_edges.csv'), index=False)
    print(f"  Saved: grn_model_edges.csv  ({len(edges)} edges, " f"{edges['source'].nunique()} TFs -> {edges['target'].nunique()} targets)")
    export_graphml(edges, tf_activity, os.path.join(OUTPUT_DIR, 'grn_model.graphml'))

    # 7. Sanity check vs the existing TF list
    if TF_LIST_CSV and os.path.exists(TF_LIST_CSV):
        known_tf_table = pd.read_csv(TF_LIST_CSV)
        name_col = _pick_col(known_tf_table.columns, ['gene_name', 'name', 'symbol'])
        known_tfs = set(str(x).upper() for x in (known_tf_table[name_col] if name_col else known_tf_table.iloc[:, 0]))
        inferred_tfs = set(str(x).upper() for x in significant_tfs.index)
        overlap = known_tfs & inferred_tfs
        print(f"\nSanity check vs {os.path.basename(TF_LIST_CSV)}: "
              f"{len(overlap)} of your known DE TFs also flagged by activity inference.")
        if overlap:
            print("  Overlap:", ", ".join(sorted(overlap))[:300])

    # 8. Reusability demonstration on a second dataset
    if SECOND_DATASET_CSV and os.path.exists(SECOND_DATASET_CSV):
        print(f"\nApplying the SAME model to {os.path.basename(SECOND_DATASET_CSV)} ...")
        deg_results_2 = pd.read_csv(SECOND_DATASET_CSV, index_col=0)
        stat_col_2 = STAT_COL if STAT_COL in deg_results_2.columns else LOG2FC_COL
        # Reuse the same orthology map where ids overlap; map any new ids too.
        new_ids = [g for g in deg_results_2.index if g not in set(ortho_map['zfish_id'])]
        ortho_map_2 = ortho_map
        if new_ids:
            ortho_map_2 = pd.concat([ortho_map, map_orthologues(new_ids, SOURCE_ORG, TARGET_ORG)], ignore_index=True)
        human_stat_2 = build_human_stat_vector(deg_results_2, ortho_map_2, stat_col_2, agg=AGG)
        tf_activity_2 = run_tf_activity(human_stat_2, collectri, label='dataset2')
        tf_activity_2.to_csv(os.path.join(OUTPUT_DIR, 'tf_activity_dataset2.csv'))
        print("  Saved: tf_activity_dataset2.csv")
        plot_tf_bar(tf_activity_2, 'dataset2', os.path.join(OUTPUT_DIR, 'tf_activity_barplot_dataset2.png'))

    print("\nProcess finished successfully! Outputs in:", OUTPUT_DIR)

if __name__ == '__main__':
    main()