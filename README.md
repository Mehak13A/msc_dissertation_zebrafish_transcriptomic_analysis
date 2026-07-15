# msc_dissertation_zebrafish_transcriptomic_analysis
Transcriptomic analysis of telomerase loss and natural ageing in zebrafish skeletal muscle: telomerase-knockout (tert -/-) vs wild-type across age. MSc Data Science dissertation, Kings College London 2026.

This repository contains the Python analysis scripts for my MSc dissertation, which investigates whether telomerase loss in zebrafish skeletal muscle reproduces natural ageing at the transcriptional level, or represents a distinct biological programme.

# Project Overview
Telomere shortening is thought to drive age-related tissue dysfunction, but whether it recapitulates normal ageing or produces a separate transcriptional signature is not well understood. This project uses bulk RNA-seq from zebrafish (Danio rerio) flank skeletal muscle to answer three core questions:

**1. Q1 Genotype effect:** what does telomerase loss (tert−/-) change at 18 months compared to wild-type?

**2. Q2 Age effect:** what changes between 11-month and 37-month wild-type muscle (natural ageing)?

**3. Q3 Overlap:** of genes significant in both, do Q1 and Q2 move in the same or opposite direction?

**4. Q4 Age-dependence:** does the Q1 genotype effect exist at 3 months (young fish), or does it emerge only with age?

A 3-month young-fish dataset (JenAge) is included as a validation cohort to answer Q4, and a genotype x age interaction model formally tests it across the combined 3-month and 18-month data.

The short answer: telomerase loss is not accelerated natural ageing. The dominant mutant feature: a coordinated E2F-driven cell-cycle shutdown is absent from the natural ageing signature, the genotype effect is entirely age-dependent (silent at 3 months, large at 18 months), and the two programmes are essentially independent.

# Repository structure (Provided in separate folders inside the Code files folder; one for each script)
scripts/
    
    1. differential_expression_analysis.py
    
    2. q2_age_outlier_sensitivity.py
    
    3. go_enrichment_analysis.py
    
    4. tert_3month_de_analysis.py
    
    5. cross_dataset_direction_check.py
    
    6. grn_tf_activity_model.py
    
    7. string_prep_and_grn_overlap.py
    
    8. q2_age_tf_activity_and_comparison.py
    
    9. interaction_genotype_x_age.py
    
    10. grn_interaction_pattern_mapping.py
    
    11. project_overview_plots.py

Each script is self-contained and documented with a full docstring explaining its purpose, inputs, outputs, and lists all the references. The scripts are numbered in the order they were run in the analysis pipeline.

# Scripts
**1. differential_expression_analysis.py**
Main DE pipeline answering Q1, Q2, Q3. Runs pydeseq2, applies Storey q-values, produces volcano plots, PCA, heatmaps and GO enrichment.

**2. q2_age_outlier_sensitivity.py**
Checks whether the Q2 ageing result holds when three PCA-flagged outlier samples are removed.

**3. go_enrichment_analysis.py**
GO/KEGG enrichment via g:Profiler on the up- and down-regulated DEG sets separately, plus Q3 overlap analysis.

**4. tert_3month_de_analysis.py**
DE analysis on the young (3-month) fish. Establishes the age-dependent phenotype: essentially no DEGs at 3 months.

**5. cross_dataset_direction_check.py**
Tests whether the 18-month DEGs show early directional drift at 3 months before reaching significance.

**6. grn_tf_activity_model.py**
Maps zebrafish genes to human orthologues, then scores all CollecTRI regulons (1,185 TFs) via decoupler to infer TF activity. Exports the GRN as an edge list and GraphML.

**7. string_prep_and_grn_overlap.py**
Prepares gene lists for the STRING protein-network enrichment analysis and cross-references STRING results against the GRN target sets.

**8. q2_age_tf_activity_and_comparison.py**
Scores all TFs independently on the ageing contrast (Q2) and compares the full mutant and ageing TF-activity tables to test the accelerated-ageing hypothesis.

**9. interaction_genotype_x_age.py**
Fits a 2x2 DESeq2 interaction model (genotype x age) on the combined 3-month and 18-month data. Classifies every gene by whether it is affected by genotype, age, both or neither, and by how its genotype effect changes with age.

**10. grn_interaction_pattern_mapping.py**
Maps GRN target genes onto the interaction classification and tests whether they are enriched for the age-dependent-onset pattern (Fisher exact test).

**11. project_overview_plots.py**
Generates summary figures: the sample design, TF counts per contrast, and a results dashboard across all analyses.

# Dependencies
All scripts run in Python 3.12. Install all required packages with:

pip install pydeseq2 decoupler omnipath gprofiler-official pandas numpy scipy matplotlib seaborn scikit-learn adjustText networkx

A note on each package and why it is needed:

**pydeseq2:** Differential expression (all DE scripts and the interaction model)

**decoupler:** TF activity inference from CollecTRI regulons (scripts 6 and 8)

**omnipath:** Downloads CollecTRI regulon network at runtime (scripts 6 and 8)

**gprofiler-official:** Gene Ontology / pathway enrichment and zebrafish-to-human orthology mapping (scripts 3 and 6)

**pandas:** Data loading and manipulation (all scripts)

**numpy:** Numerical operations (all scripts)

**scipy:** Storey q-values, Fisher exact test, statistical utilities (all scripts)

**matplotlib:** All figures (all scripts)

**seaborn:** Heatmaps and distribution plots (scripts 1, 2, 3)

**scikit-learn:** PCA computation and dimensionality reduction (scripts 1 and 2)

**adjustText:** Automatically repositions overlapping text labels on scatter plots and volcano plots (scripts 1, 3, 5)

**networkx:** GRN graph construction and GraphML export (script 6)

Scripts 6 and 8 require an internet connection to download the CollecTRI regulons via OmniPath. All other scripts run fully offline once the data files are in place.

For the STRING analysis (script 7), the gene lists are submitted to the STRING website (string-db.org) manually. The script handles input preparation and downstream cross-referencing only, and does not require a STRING account or API key.

# Data
The raw count datasets are provided in separately in this repository:

    adult_muscle_18mo/
        gene_count.xls: 18-month adult muscle raw counts (Novogene); Groups: M18_MUT (n=4), M18_WT (n=5),M37_WT (n=6), M11_WT (n=5)

    tert3m_raw_counts.csv: Combined 3-month count matrix, produced by script 4 (tert_3month_de_analysis.py) from 13 individual featureCounts
    files (one per sample). The raw featureCounts files are not included due to file size; this combined matrix is the input used by all
    downstream scripts. Groups: M3_MUT (n=3), M3_WT (n=5), M3_Het (n=5)

# Outputs
1. differential_expression_analysis/
2. q2_age_outlier_sensitivity/
3. go_enrichment_analysis/
4. tert_3month_de_analysis/
5. cross_dataset_direction_check/
6. grn_tf_activity_model/
7. string_prep_and_grn_overlap/
8. q2_age_tf_activity_and_comparison/
9. interaction_genotype_x_age/
10. grn_interaction_pattern_mapping/
11. project_overview_plots/

Outputs Explanation_date.docx: plain-language description of every output file

# How to Run
1. Clone this repository.
2. Install the dependencies as stated before.
3. The raw data files are provided seprately as mentioned before. Open whichever script you want to run and set the paths in the CONFIGURATION block at the top to point to those files.
4. Run the scripts in order (1 through 11), as later scripts use outputs from earlier ones. Each script's docstring lists exactly which input files it needs and which output files it produces. The pre-computed outputs are already provided seprately in folders mentioned before if you want to inspect results without re-running the analysis.
