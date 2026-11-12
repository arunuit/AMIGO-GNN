# Data preparation schema

The manuscript uses CARD, BV-BRC and NCBI Pathogen Detection. The exact source snapshot is not encoded in the manuscript, so this repository separates the software from the data preparation.

## Required files

### samples.csv

| column | meaning |
|---|---|
| sample_id | unique isolate/accession identifier |
| label | Resistant / Intermediate / Susceptible |
| species | bacterial species |

### nodes.csv

| column | meaning |
|---|---|
| node_id | unique biological entity |
| node_type | one of the six node categories |
| f0...fN | numeric node features |

### edges.csv

| column | meaning |
|---|---|
| source | source node ID |
| target | target node ID |
| relation | semantic edge type |

Recommended relation vocabulary from the manuscript:

- has_gene
- contains_mutation
- resistant_to
- associated_with
- participates_in

### sample_nodes.csv

Maps each isolate/sample to the nodes belonging to its graph.

## Deduplication

Perform deduplication using accession/isolate identifiers before splitting. If multiple records represent the same biological isolate, keep one according to a documented rule.

## Split

Use 70% train, 15% validation, 15% test as reported in the manuscript. For a stronger generalization analysis, add a strain-/lineage-aware split and report it separately.

## Leakage

Do not use test labels, susceptibility values or test-derived statistics when fitting normalization, BioMOS masks, or model parameters.
