# Split dataset workflow

This workflow generates validated train/validation/test partitions from non-reduced and reduced sequence datasets. It is designed to produce fixed split assignments that can be reused across numerical representations and downstream machine-learning experiments.

```text
endpoint-specific sequence data
reduced datasets (optional)
numerical representations (distance-aware strategies only)
                      ↓
                 split_dataset
                      ↓
            split_process/<dataset>/
```

The permanent fold files contain only the configured identifier and label columns. Numerical features are stored separately and can be joined to the split assignments when training a model.

---

## 1. Workflow behavior

The workflow supports two types of split scenarios.

### Representation-independent scenarios

The following strategies depend only on the sequence identifiers and labels in a dataset variant:

- `random_kfold`
- `stratified_kfold`

These partitions are generated once per dataset variant, strategy, seed, and reduction level. They are not repeated for every numerical representation.

### Representation-dependent scenarios

The following strategy uses numerical features to construct the partitions:

- `distance_aware_kfold`

Distance-aware partitions are generated separately for each representation selected in `cross_features.train_representations`.

All valid fold files are compacted after generation so that the permanent `train.csv`, `val.csv`, and `test.csv` files contain only:

```text
<id_col>,<label_col>
```

With the provided configuration, these columns are:

```text
id,label
```

---

## 2. Workflow location

The workflow directory is expected to contain:

```text
pipelines/split_dataset/
├── Snakefile
├── README.md
└── config/
    └── config.yaml
```

Run the commands from `pipelines/split_dataset/` so that relative paths in the configuration resolve correctly.

---

## 3. Requirements

The workflow requires:

- Python;
- Snakemake;
- pandas;
- PyYAML;
- NumPy for distance-aware strategies;
- BioSieve available through the command configured in `split_strategies.biosieve_exec`.

The random-seed CSV must contain a column named:

```text
seed
```

---

## 4. Input data

### 4.1 Endpoint-specific source dataset

The main source is configured through:

```yaml
dataset:
  input_data: "../data/toxic/sequences.csv"
```

For `random_kfold` and `stratified_kfold`, the source must contain:

```text
id
label
```

The exact names are controlled by `dataset.id_col` and `dataset.label_col`.

For `distance_aware_kfold`, the source must additionally contain the configured sequence column:

```text
sequence
```

Identifiers must not have conflicting labels. Null identifiers and labels are rejected before split generation.

### 4.2 Numerical representations

Numerical representations are required only when `distance_aware_kfold` is enabled. Each representation is loaded from:

```text
<representation_root>/
└── <representation_dataset>/
    └── <method>/
        └── <model_alias>/
            └── full_data.csv
```

The representation file must contain either the configured identifier column or the configured sequence column so that it can be joined to the endpoint-specific source dataset.

### 4.3 Reduced datasets

Optional reduced datasets can be loaded from distance-, descriptor-, or homology-reduction outputs.

For distance and descriptor reductions, the workflow searches each reduction-level directory in this order:

```text
data_nr_labeled.csv
data_nr.csv
```

For homology reductions, it searches:

```text
data_nr_labeled.csv
data_nr_mmseqs2.csv
data_nr.csv
```

At least one source in `split_sources` must be enabled.

---

## 5. Running the workflow

From the workflow directory:

```bash
cd pipelines/split_dataset

python -m snakemake \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

Preview the planned jobs without executing them:

```bash
python -m snakemake \
  --dry-run \
  --printshellcmds
```

The Snakefile loads `config/config.yaml` by default. A different configuration can be supplied explicitly:

```bash
python -m snakemake \
  --snakefile Snakefile \
  --configfile path/to/config.yaml \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  --printshellcmds
```

---

## 6. Configuration reference

### 6.1 `global`

```yaml
global:
  output_root: "../.."
  representation_root: "../../numerical_representation_data"
  reduction_root: "../.."
  seeds_file: "../../general_configs/random_seeds_30.csv"
```

| Key | Description |
|---|---|
| `output_root` | Base project path used for default output locations. |
| `representation_root` | Root containing numerical-representation datasets. |
| `reduction_root` | Base path used when a reduction root is not provided explicitly. |
| `seeds_file` | CSV containing the random seeds in a `seed` column. |

Relative paths are resolved from the directory where Snakemake is executed.

### 6.2 `dataset`

```yaml
dataset:
  name: "maomao_toxic"
  representation_dataset: "maomao"
  input_data: "../data/toxic/sequences.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

| Key | Description |
|---|---|
| `name` | Dataset name used in permanent and temporary output paths. |
| `representation_dataset` | Dataset folder used under `representation_root`. Defaults to `dataset.name` when omitted. |
| `input_data` | Endpoint-specific source CSV. |
| `sequence_col` | Peptide or biological-sequence column. |
| `id_col` | Unique identifier used by BioSieve and permanent split files. |
| `label_col` | Target-label column. |

`representation_dataset` allows several endpoint datasets to reuse numerical representations generated for a shared sequence collection.

### 6.3 `representations`

```yaml
representations:
  ankh2_ext1:
    method: "sylphy_embedding"
    model_alias: "ankh2_ext1"
    output_name: "ankh2_ext1"
    feature_mode: "embeddings"
    metric: "cosine"

  one_hot:
    method: "sylphy_one_hot"
    model_alias: "one_hot"
    output_name: "one_hot"
    feature_mode: "descriptors"
    metric: "euclidean"
    feature_prefix: "p_"
```

| Key | Description |
|---|---|
| `method` | Directory containing the representation model output. |
| `model_alias` | Model directory containing `full_data.csv`. |
| `output_name` | Filesystem-safe representation name used in distance-aware scenario folders. |
| `feature_mode` | Either `embeddings` or `descriptors`. |
| `metric` | Distance metric passed to BioSieve. |
| `feature_prefix` | Prefix used to identify descriptor columns. Required for descriptor-based distance-aware splits. |

Representation definitions do not cause `random_kfold` or `stratified_kfold` to be repeated. They are used only by enabled representation-dependent strategies.

### 6.4 `split_sources`

```yaml
split_sources:
  no_reduced:
    enabled: true

  reduced_distance:
    enabled: false
    reductions:
      esm2_t6_8M_UR50D:
        enabled: false
        representation_key: "esm2_t6_8M_UR50D"
        root: "../../reduced_distance/<dataset>/<reduction>"
        thresholds: "auto"

  reduced_homology:
    enabled: false
    reductions:
      homology_mmseqs2_reduction:
        enabled: false
        root: "../../reduced_homology/<dataset>/<reduction>"
        thresholds: "auto"

  reduced_descriptor:
    enabled: false
    reductions:
      one_hot:
        enabled: false
        representation_key: "one_hot"
        root: "../../reduced_descriptor/<dataset>/<reduction>"
        thresholds: "auto"
```

#### Non-reduced source

When enabled, `no_reduced` uses `dataset.input_data` directly.

#### Distance-reduced source

`representation_key` identifies the representation used to create the reduced sequence universe. It must match a key in `representations`.

#### Descriptor-reduced source

`representation_key` identifies the descriptor representation used to create the reduced sequence universe.

#### Homology-reduced source

The reduction name is used to distinguish different homology-reduction variants.

#### Threshold selection

When `thresholds` is set to `auto`, the workflow detects directories beginning with:

```text
p
threshold_
```

Examples include:

```text
p70_0
p99_5
threshold_0.7
```

A specific list can also be provided:

```yaml
thresholds:
  - "p70_0"
  - "p90_0"
  - "p99_5"
```

### 6.5 `split_strategies`

```yaml
split_strategies:
  biosieve_exec: "biosieve"

  random_kfold:
    enabled: true
    n_splits: 5
    shuffle: true
    val_size: 0.1

  stratified_kfold:
    enabled: true
    n_splits: 5
    shuffle: true
    val_size: 0.1
    dropna: true
    cast_to_str: false

  distance_aware_kfold:
    enabled: false
    n_splits: 5
    val_size: 0.1
    shuffle_ties: true
    descriptor_modes:
      - "no_norm"
```

| Strategy | Dependency |
|---|---|
| `random_kfold` | Sequence/label universe only. |
| `stratified_kfold` | Sequence/label universe and class labels. |
| `distance_aware_kfold` | Sequence/label universe and one selected numerical representation. |

For descriptor representations, `descriptor_modes` controls whether descriptor features are standardized before distance-aware partitioning:

```yaml
descriptor_modes:
  - "no_norm"
  - "norm"
```

These modes produce separate strategy directories:

```text
distance_aware_kfold_no_norm/
distance_aware_kfold_norm/
```

Embedding representations use the configured distance-aware output name, which defaults to:

```text
distance_aware_kfold/
```

### 6.6 `cross_features`

```yaml
cross_features:
  train_representations:
    - "ankh2_ext1"
    - "esm2_t6_8M_UR50D"
    - "one_hot"
```

Despite the configuration-key name, this block does not materialize every random or stratified split with every representation.

It defines the representations used to generate `distance_aware_kfold` scenarios. When distance-aware splitting is disabled, this list does not create additional split jobs.

Every value must match a key in `representations`.

### 6.7 `output`

```yaml
output:
  root: "../../split_process"
  include_dataset_folder: true
  materialized_root: "../../split_process_inputs/maomao_toxic"
```

| Key | Description |
|---|---|
| `root` | Root for permanent split outputs. |
| `include_dataset_folder` | When `true`, places outputs under `<root>/<dataset.name>/`. |
| `materialized_root` | Temporary datasets and distance arrays used during split generation. |

### 6.8 `validation`

```yaml
validation:
  enabled: true
  min_classes: 2
  min_classes_per_split: 2
  required_split_files:
    - "train.csv"
    - "val.csv"
    - "test.csv"
  remove_stale_outputs_before_split: true
  keep_invalid_fold_files: false
  remove_invalid_run_dirs: true
```

The workflow validates inputs before calling BioSieve and validates generated folds afterward.

Checks include:

- required identifier and label columns;
- null identifiers and labels;
- conflicting labels for the same identifier;
- at least the configured number of classes in the source dataset;
- enough rows for the configured number of folds;
- enough samples per class for stratified splitting;
- presence and non-empty content of required fold files;
- minimum class diversity in every train, validation, and test subset;
- valid identifier and label columns before permanent split files are written.

### 6.9 `analysis`

```yaml
analysis:
  enabled: false
  script: "../../notebooks_and_scripts/scripts_for_pipelines/split_summary.py"
  output_dir: null
  summary_dirname: "split_analysis"
  plots:
    fig_format: "png"
    dpi: 300
    cmap: "split_feasibility"
    annotate: true
    include_no_threshold: true
```

When enabled, the configured script receives the split root, dataset name, output directory, and plotting options.

If `output_dir` is `null`, outputs are written to:

```text
split_process/<dataset>/split_analysis/
```

The workflow records completion through:

```text
analysis.done
```

The exact tables and figures depend on the configured analysis script.

### 6.10 `cleanup`

```yaml
cleanup:
  remove_materialized_inputs: true
```

When enabled, the workflow removes:

```text
split_process_inputs/<dataset>/
```

after all split summaries and the optional analysis stage are complete. A marker is then written to the permanent output area:

```text
.materialized_inputs_removed.done
```

The parent `split_process_inputs/` directory is removed only when it becomes empty.

---

## 7. Scenario naming and output organization

The permanent root is determined by:

```text
output.root[/dataset.name]
```

when `include_dataset_folder` is enabled.

### 7.1 Non-reduced random and stratified splits

These scenarios do not include a representation name:

```text
split_process/<dataset>/
└── no_reduced/
    ├── random_kfold/
    │   └── seed_<seed>/
    └── stratified_kfold/
        └── seed_<seed>/
```

### 7.2 Reduced random and stratified splits

Reduced variants remain distinct because each reduction can contain a different sequence universe:

```text
split_process/<dataset>/
├── reduced_distance_by_<reduction_representation>/
│   └── <strategy>/
│       └── seed_<seed>/
│           └── <threshold>/
├── reduced_descriptor_by_<reduction_representation>/
│   └── <strategy>/
│       └── seed_<seed>/
│           └── <threshold>/
└── reduced_homology_<reduction_name>/
    └── <strategy>/
        └── seed_<seed>/
            └── <threshold>/
```

### 7.3 Distance-aware splits

Distance-aware scenarios include the representation used to calculate distances:

```text
split_process/<dataset>/
└── <representation>_no_reduced/
    └── distance_aware_kfold/
        └── seed_<seed>/
```

Reduced distance-aware scenarios additionally identify the reduction source and threshold.

### 7.4 Seed and fold structure

A valid no-threshold seed directory contains:

```text
seed_<seed>/
├── biosieve_split.stderr.log
├── biosieve_split.stdout.log
├── DONE.txt
├── kfold_report.json
├── params_split.yaml
├── split_summary.csv
├── fold_00/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── fold_01/
├── fold_02/
├── fold_03/
└── fold_04/
```

For reduced datasets, the fold directories and run-specific files are placed inside the reduction-level directory:

```text
seed_<seed>/
├── split_summary.csv
└── <threshold>/
    ├── DONE.txt
    ├── kfold_report.json
    ├── params_split.yaml
    ├── biosieve_split.stdout.log
    ├── biosieve_split.stderr.log
    └── fold_00/ ... fold_04/
```

`split_summary.csv` remains at seed level and contains one row per attempted reduction level.

---

## 8. Permanent file contents

Each valid fold contains:

| File | Role |
|---|---|
| `train.csv` | Training assignment for the fold. |
| `val.csv` | Validation assignment for the fold. |
| `test.csv` | Held-out test assignment for the fold. |

After validation, each file is rewritten atomically and contains only:

```csv
id,label
```

The exact header follows `dataset.id_col` and `dataset.label_col`.

Additional seed- or run-level files include:

| File | Description |
|---|---|
| `params_split.yaml` | BioSieve parameters used for the split. |
| `kfold_report.json` | Report generated by BioSieve. |
| `split_summary.csv` | Public status table for the attempted reduction levels. |
| `biosieve_split.stdout.log` | Captured BioSieve standard output. |
| `biosieve_split.stderr.log` | Captured BioSieve standard error. |
| `DONE.txt` | Contains `kept` or `invalid_split`. |
| `INVALID_SPLIT.txt` | Reason for an invalid no-threshold split. |

---

## 9. Reconstructing numerical datasets for training

Random and stratified split files contain assignments rather than duplicated numerical features. Reconstruct a training dataset by joining a fold file with the required representation using the identifier column.

```python
import pandas as pd

split_df = pd.read_csv(
    "../../split_process/maomao_toxic/no_reduced/"
    "stratified_kfold/seed_113/fold_00/train.csv"
)

representation_df = pd.read_csv(
    "../../numerical_representation_data/maomao/"
    "sylphy_embedding/ankh2_ext1/full_data.csv"
).drop(columns=["label"], errors="ignore")

train_df = split_df.merge(
    representation_df,
    on="id",
    how="inner",
)
```

The label from the split file should remain authoritative. The same split assignments can therefore be reused with every compatible numerical representation.

---

## 10. Split status and invalid runs

Each split attempt writes an internal status row. The rows are combined into the public seed-level:

```text
split_summary.csv
```

For a non-reduced scenario, the table has the form:

```csv
reduction_levels,status,reason
no_threshold,kept,
```

For percentile reductions:

```csv
percentile,reduction_levels,status,reason
70.0,p70_0,kept,
90.0,p90_0,invalid_split,<reason>
```

For homology thresholds:

```csv
min_seq_id,reduction_levels,status,reason
0.4,threshold_0.4,kept,
```

Only runs with:

```text
status == kept
```

should be used downstream.

When invalid fold files are not retained:

- stale fold outputs are removed before rerunning;
- invalid fold directories are removed;
- invalid threshold directories can be removed completely;
- the failure reason remains available in the seed-level summary.

---

## 11. Behavior of the provided configuration

The provided `config.yaml` currently enables:

```text
source: no_reduced
strategies: random_kfold, stratified_kfold
analysis: disabled
cleanup of materialized inputs: enabled
```

The reduced sources and `distance_aware_kfold` are currently disabled.

Therefore, the active permanent output structure is:

```text
split_process/
└── maomao_toxic/
    └── no_reduced/
        ├── random_kfold/
        │   └── seed_<seed>/
        │       └── fold_00/ ... fold_04/
        └── stratified_kfold/
            └── seed_<seed>/
                └── fold_00/ ... fold_04/
```

Although 11 representations are listed in the configuration, they do not duplicate these random or stratified partitions. They become active for split generation only when `distance_aware_kfold` is enabled.

---

## 12. Adapting the workflow to another dataset

1. Prepare a source CSV with the configured identifier and label columns. Include the sequence column when distance-aware splitting is required.
2. Update the `dataset` block.
3. Set `representation_dataset` when the representation folder differs from `dataset.name`.
4. Define any representations required for distance-aware splitting.
5. Enable the required source variants.
6. Update reduction roots and threshold selections when reduced datasets are used.
7. Choose the split strategies.
8. Set a dataset-specific temporary `materialized_root`.
9. Run a dry-run before executing the workflow.

Example:

```yaml
dataset:
  name: "my_endpoint"
  representation_dataset: "shared_sequence_collection"
  input_data: "../data/my_endpoint/sequences.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"

output:
  root: "../../split_process"
  include_dataset_folder: true
  materialized_root: "../../split_process_inputs/my_endpoint"
```

Then run:

```bash
python -m snakemake \
  --dry-run \
  --printshellcmds

python -m snakemake \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  --printshellcmds
```

---

## 13. Downstream use

Downstream workflows should:

1. select only split attempts marked `kept`;
2. preserve the provided seed and fold assignments;
3. load the required `train.csv`, `val.csv`, and `test.csv` files;
4. join the selected numerical representation by the configured identifier;
5. avoid regenerating random or stratified partitions independently for each representation.

This preserves paired comparisons among numerical representations and prevents data-assignment differences from confounding model evaluation.
