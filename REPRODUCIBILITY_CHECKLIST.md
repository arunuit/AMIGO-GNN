# Reproducibility checklist

Before submitting the revised manuscript, archive the following:

- [ ] exact CARD release/version
- [ ] exact BV-BRC release/version
- [ ] exact NCBI Pathogen Detection snapshot/date
- [ ] download URLs or accession identifiers
- [ ] deduplication script and rule
- [ ] ontology/entity mapping file
- [ ] final isolate list
- [ ] train/validation/test isolate IDs
- [ ] node list and edge list
- [ ] feature matrix
- [ ] BioMOS random seed
- [ ] GAT random seed(s)
- [ ] number of repeated seeds
- [ ] mean ± standard deviation across runs
- [ ] final confusion matrix
- [ ] metrics calculated from the same predictions as the matrix
- [ ] ROC/AUC and PR/AUPRC calculated from the same predictions
- [ ] SHAP explainer type and parameters
- [ ] GNNExplainer parameters
- [ ] perturbation-fidelity definition and code
- [ ] dashboard code
- [ ] data/code availability statement
- [ ] environment/package versions


## Manuscript-specific release gate

- [ ] Archive the exact processed cohort (or a legally distributable manifest with stable accession IDs and checksums).
- [ ] Archive the exact stratified and lineage-aware split IDs used for every reported run.
- [ ] Supply the source-versioned CARD, BV-BRC and NCBI preprocessing and identifier-mapping records.
- [ ] Supply the finalized BGKG and document provenance for every relation, especially `resistant_to`.
- [ ] Confirm the BioMOS graph corresponds to the biological feature mapping used in the manuscript; the generic correlation-graph fallback is not sufficient evidence of that claim.
- [ ] Regenerate every manuscript table and figure from saved predictions and per-run outputs.
- [ ] Reconcile the confusion matrix with the reported 98.82% accuracy and MCC 0.976.
