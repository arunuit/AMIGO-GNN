# Dataset partitions

Store the exact sample IDs used for each reported run. `scripts/train.py` exports its generated stratified assignments to each run directory. For lineage-aware partitions, add a `lineage_group` column (MLST or SNP cluster) to `samples.csv` and run `python scripts/make_lineage_split.py`. The script writes group-disjoint assignments; verify class balance and report the realized proportions before using it in the manuscript. It does not recreate historical partitions unless the same source table, groups, and seed are supplied.
