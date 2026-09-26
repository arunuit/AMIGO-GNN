# AMIGO-GNN: Reproducible Implementation

This repository is an implementation-oriented companion for the manuscript:

**A Multi-Objective Evolutionary Graph Neural Framework for Interpretable Antimicrobial Resistance Prediction**

The manuscript describes AMIGO-GNN as a pipeline containing:

1. public AMR/genomic data integration,
2. a heterogeneous bacterial genomic knowledge graph (BGKG),
3. BioMOS graph-aware multi-objective feature optimization,
4. GAT graph representation learning,
5. BioReLU activation,
6. SHAP feature attribution,
7. GNNExplainer graph explanations,
8. quantitative explanation-fidelity testing, and
9. an optional Streamlit dashboard.

The manuscript reports Python 3.11, PyTorch/PyTorch Geometric, Adam, learning rate 0.001, hidden dimension 128, 8 attention heads, 2 GAT layers, dropout 0.5, 200 epochs, early stopping patience 20, BioMOS population 50, 100 iterations, objective weights (0.40, 0.25, 0.20, 0.15), graph-awareness coefficient 0.30, and BioReLU sensitivity 0.10.

## Important reproducibility note

The manuscript names CARD, BV-BRC and NCBI Pathogen Detection as public sources, but it does **not** provide a machine-readable snapshot of the exact records, identifiers, release dates, final graph, or feature matrix used for the reported experiment. Therefore this repository does not fabricate those missing inputs.

Instead it provides:

- a complete runnable software pipeline;
- a documented CSV schema for importing the real data;
- a synthetic smoke-test dataset;
- paper-metric verification from the published confusion matrix;
- deterministic seeds;
- leakage-safe split-before-graph construction;
- BioMOS implementation;
- GAT + BioReLU;
- SHAP KernelSHAP wrapper;
- GNNExplainer;
- perturbation-based explanation fidelity;
- metric/ROC/PR/confusion-matrix plots;
- a Streamlit dashboard.

Use the synthetic dataset only to test the software. Do **not** use its results as experimental results in the paper.

## 1. Environment

Recommended:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

For CUDA-specific PyTorch installations, install the appropriate PyTorch build first from the official PyTorch installation selector, then install the remaining requirements.

## 2. Smoke test

Generate a small graph dataset:

```bash
python scripts/make_demo_data.py
```

Train the model:

```bash
python scripts/train.py --config configs/demo.yaml
```

Evaluate:

```bash
python scripts/evaluate.py --run_dir runs/demo
```

Run explanation:

```bash
python scripts/explain.py --run_dir runs/demo
```

Run perturbation fidelity:

```bash
python scripts/fidelity.py --run_dir runs/demo --samples 50
```

Launch the dashboard:

```bash
streamlit run dashboard.py
```

## 3. Real-data input format

The code uses four files:

### `samples.csv`

```text
sample_id,label,species
iso_0001,Resistant,Escherichia coli
iso_0002,Susceptible,Escherichia coli
```

### `nodes.csv`

```text
node_id,node_type
n1,Bacterial Species
n2,Resistance Gene
n3,Antibiotic
n4,Resistance Mechanism
n5,Biological Pathway
```

Additional numeric columns are interpreted as node features, for example:

```text
node_id,node_type,f0,f1,f2,f3
n1,Bacterial Species,1,0,0.4,0
n2,Resistance Gene,0,1,0.9,1
```

### `edges.csv`

```text
source,target,relation
n1,n2,has_gene
n2,n3,resistant_to
n3,n4,associated_with
n4,n5,participates_in
```

### `sample_nodes.csv`

```text
sample_id,node_id
iso_0001,n1
iso_0001,n2
iso_0001,n3
iso_0001,n4
iso_0001,n5
```

The sample split is performed **before** sample-specific graph assembly so that test isolates do not influence training graph construction.

## 4. Converting real CARD/BV-BRC/NCBI data

The manuscript identifies the following sources:

- CARD: resistance genes, ARO, resistance mechanisms, drug classes and AMR phenotypes.
- BV-BRC: bacterial genomes, AMR genes, susceptibility and genome annotations.
- NCBI Pathogen Detection: whole-genome sequences, SNPs, AMR genes and isolate metadata.

Because the manuscript does not specify the exact downloaded release and record IDs, the repository does not silently download a particular snapshot and call it the original experiment. Prepare the four CSV files above from the exact data snapshot you use and record:

- source release/version,
- download date,
- accession/record identifiers,
- deduplication rule,
- ontology mapping version,
- train/validation/test IDs.

## 5. Leakage control

For a defensible experiment:

1. deduplicate at isolate/accession level;
2. split isolates into train/validation/test;
3. fit feature normalization using training data only;
4. construct sample graphs after the split;
5. fit BioMOS using training/validation data only;
6. freeze the selected feature mask before test evaluation;
7. evaluate the final model once on the held-out test set.

If ontology/reference edges are used globally, document which edges are reference knowledge and which are sample-derived. Do not allow test-isolate labels or susceptibility measurements to enter the training graph.

## 6. BioMOS

The implementation follows the manuscript's stated objective:

`F = alpha*A + beta*Gc + gamma*R - delta*D`

where:

- `A` = validation prediction score,
- `Gc` = graph-connectivity score,
- `R` = biological relevance score,
- `D` = feature redundancy.

Default weights are:

`alpha=0.40, beta=0.25, gamma=0.20, delta=0.15`

The optimizer uses binary feature masks, local/global best solutions and graph-topology guidance.

## 7. BioReLU

The manuscript specifies:

```text
BioReLU(x) = x                         if x >= 0
             lambda*x*exp(x)           otherwise
```

with `lambda=0.10`.

The implementation is numerically stabilized with a clamp on the negative branch.

## 8. SHAP method

The repository uses **KernelSHAP** through `shap.KernelExplainer` around a graph-model prediction wrapper. This is intentionally explicit because a generic SHAP call is not sufficient to establish which approximation was used.

The wrapper masks input feature dimensions while keeping the graph topology fixed.

For large feature spaces, use a small background set and limit `nsamples`. For a publication-grade experiment, report:

- explainer: KernelSHAP,
- background size,
- number of explained samples,
- `nsamples`,
- feature masking rule.

## 9. Explanation fidelity

The repository includes a perturbation-based fidelity test.

For an explained graph, the top-k explanation features are retained while non-selected feature dimensions are masked. Fidelity is:

```text
Fidelity =
(1/N) * sum_i I[f(G_i^E) = f(G_i)] * 100
```

where `G_i^E` is the explanation-preserving perturbed graph.

This should be reported as an actual measured value from the run. The repository does not hard-code the manuscript's 96.8% explanation-fidelity number.

## 10. Critical metric check from the manuscript

The manuscript's Figure 6 confusion matrix is:

```text
[[196, 3, 1],
 [  4,191, 5],
 [  2,  4,194]]
```

It contains 600 samples and 581 correct predictions, giving:

- accuracy = 96.8333%
- macro precision ≈ 96.8314%
- macro recall = 96.8333%
- macro F1 ≈ 96.8308%
- multiclass MCC ≈ 0.9525

Run:

```bash
python scripts/verify_paper_metrics.py
```

This check is included specifically to prevent a confusion matrix from being displayed together with incompatible headline metrics. The final manuscript metrics should be regenerated from the same `y_true` and `y_pred` arrays used to create the confusion matrix.

## 11. GitHub

```bash
git init
git add .
git commit -m "Initial AMIGO-GNN reproducible implementation"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/AMIGO-GNN.git
git push -u origin main
```

Do not upload private/raw patient or restricted genomic data. Upload only public-data processing scripts, metadata, licenses, hashes and permitted derived artifacts.

## 12. Suggested paper data/code statement

> Code for the AMIGO-GNN framework, including BioMOS optimization, GAT-BioReLU learning, SHAP-based feature attribution, GNNExplainer visualization, perturbation-based explanation-fidelity evaluation, metric verification and dashboard components, is provided in the accompanying repository. The exact public-data snapshot and preprocessing metadata should be archived with accession identifiers and version information to support full reproducibility.

