# Biological knowledge graph construction

Document the exact node/edge schemas, source provenance, identifier normalization, deduplication, and edge-generation rules used for the manuscript graph. In particular, record whether `resistant_to` edges come from curated reference knowledge or isolate-specific AST labels. The generic CSV graph loader in `src/data.py` is not by itself a CARD/BV-BRC/NCBI integration pipeline.
