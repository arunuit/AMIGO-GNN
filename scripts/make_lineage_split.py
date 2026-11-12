"""Create reproducible group-disjoint train/validation/test IDs.

Input CSV must contain sample_id, label and a lineage_group column (e.g. MLST
sequence type or SNP cluster). This script does not infer lineage assignments.
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='data/raw/samples.csv')
    ap.add_argument('--group_column', default='lineage_group')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='partitions/lineage_aware_split_ids.csv')
    args = ap.parse_args()
    df = pd.read_csv(args.samples)
    required = {'sample_id', 'label', args.group_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df[args.group_column].isna().any():
        raise ValueError('Lineage group contains missing values; resolve them explicitly.')
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=args.seed)
    train_idx, hold_idx = next(outer.split(df, df['label'], groups=df[args.group_column]))
    train, hold = df.iloc[train_idx], df.iloc[hold_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=args.seed)
    val_rel, test_rel = next(inner.split(hold, hold['label'], groups=hold[args.group_column]))
    parts = {'train': train, 'val': hold.iloc[val_rel], 'test': hold.iloc[test_rel]}
    rows = []
    for split, part in parts.items():
        rows.extend({'split': split, 'sample_id': sid, 'lineage_group': group}
                    for sid, group in zip(part.sample_id, part[args.group_column]))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Wrote {len(rows)} assignments to {out}')
    for split, part in parts.items():
        print(split, 'isolates=', len(part), 'groups=', part[args.group_column].nunique())

if __name__ == '__main__':
    main()
