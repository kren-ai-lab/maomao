# MAOMAO: An Ontology-Guided FAIR Resource for Harmonized Peptide Toxicity Data

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Snakemake](https://img.shields.io/badge/Snakemake-9.x-green.svg)](https://snakemake.github.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21584414.svg)](https://doi.org/10.5281/zenodo.21584414)

Nicole Soto-García<sup>1</sup>, Roberto Uribe-Paredes<sup>1</sup>, Leandro Murgas-Saavedra<sup>1</sup>, Karen Oróstica<sup>2</sup>, Jorge González-Puelma<sup>3,4</sup>, Marcelo Navarrete<sup>3,4</sup>, Frederic Cadet<sup>5</sup>, and David Medina-Ortiz<sup>1,*</sup>.<br>

<sup>1</sup><sub>Departamento de Ingeniería en Computación, Universidad de Magallanes, Avenida Bulnes 01855, 6210427, Punta Arenas, Chile.</sub><br>
<sup>2</sup><sub>Data Science Institute, Universidad del Desarrollo, Av. Plaza 680, 7610615, Santiago, Chile.</sub><br>
<sup>3</sup><sub>Centro Asistencial Docente e Investigación, Universidad de Magallanes, Av. Los Flamencos 01364, Punta Arenas, Chile.</sub><br>
<sup>4</sup><sub>Escuela de Medicina, Universidad de Magallanes, Avenida Bulnes 01855, Punta Arenas, Chile.</sub><br>
<sup>5</sup><sub>PEACCEL, AI for Biologics, 75013, Paris, France.</sub><br>
<sup>*</sup><sub>Corresponding author: David Medina-Ortiz ([david.medina@umag.cl](mailto:david.medina@umag.cl)).</sub><br>

---

**MAOMAO** (**M**etadata-**A**ware **O**ntology for **M**ulti-source **A**nnotation **O**rganization) is an ontology-guided resource for integrating, harmonizing, and documenting peptide toxicity annotations collected from heterogeneous public sources.

The resource follows FAIR principles—**Findability, Accessibility, Interoperability, and Reusability**—through standardized sequence identifiers, controlled endpoint terminology, explicit evidence states, source-level provenance, structured metadata, reproducible processing notebooks, and configuration-driven computational workflows.

MAOMAO is a **data resource and resource-construction framework**. It is not presented as a toxicity predictor. Its primary purpose is to provide harmonized, provenance-aware, uncertainty-aware, and computationally reusable peptide toxicity data.

---

# Table of contents

- [Overview](#overview)
- [Resource scope](#resource-scope)
- [Ontology and evidence model](#ontology-and-evidence-model)
- [Main resource outputs](#main-resource-outputs)
- [Repository structure](#repository-structure)
- [Software requirements](#software-requirements)
- [Installation](#installation)
- [Reconstructing MAOMAO](#reconstructing-maomao)
- [Computational workflows](#computational-workflows)
  - [Numerical representations](#1-numerical-representations)
  - [Endpoint-specific split preparation](#2-endpoint-specific-split-preparation)
  - [Dataset splitting](#3-dataset-splitting)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Output structure](#output-structure)
- [Reproducibility and traceability](#reproducibility-and-traceability)
- [Citation](#citation)
- [License](#license)
- [Authors and contact](#authors-and-contact)

---

# Overview

Peptide toxicity information is distributed across databases, prediction resources, supplementary datasets, and literature-derived collections. These sources frequently differ in sequence formatting, endpoint terminology, evidence interpretation, metadata completeness, and class definitions.

MAOMAO addresses these limitations through a reproducible workflow that:

- collects peptide toxicity data from multiple sources;
- standardizes and validates peptide sequences;
- harmonizes toxicity terminology;
- organizes endpoints using an explicit hierarchy;
- distinguishes positive, negative, ambiguous and unlabeled;
- preserves source-level provenance and endpoint-specific metadata;
- records hierarchy-derived annotation changes;
- produces a sequence-level pivote resource;
- generates numerical representations for computational reuse;
- creates reproducible endpoint-specific dataset partitions.

The final resource is organized around stable sequence identifiers, explicit endpoint columns, structured metadata, audit tables, and reusable workflow outputs.

---

# Resource scope

MAOMAO currently organizes evidence for the following final toxicity endpoints:

| Endpoint | Description |
|---|---|
| `toxic` | General evidence that a peptide is toxic. |
| `cytotoxic` | Toxicity affecting cells. |
| `hemolytic` | Lysis or damage of red blood cells. |
| `cytolysis` | Evidence associated with cell lysis. |
| `neurotoxic` | Toxicity affecting the nervous system. |
| `embryotoxic` | Toxicity affecting embryos or embryonic development. |
| `ichthyotoxic` | Toxicity affecting fish. |

Source-specific terminology is normalized during processing so that heterogeneous annotations can be integrated into this common endpoint vocabulary.

---

# Ontology and evidence model

## Endpoint hierarchy

MAOMAO uses the following toxicity hierarchy:

```text
Toxic
├── Cytotoxic
│   ├── Hemolytic
│   └── Cytolysis
├── Neurotoxic
├── Embryotoxic
└── Ichthyotoxic
```

## Evidence encoding

The sequence-level pivot uses the following codes:

| Code | Evidence state | Meaning |
|---:|---|---|
| `0` | Negative | Explicit negative evidence for the endpoint. |
| `1` | Positive | Explicit or hierarchy-supported positive evidence. |
| `2` | Ambiguous | Conflicting or unresolved evidence. |
| `3` | Unlabeled | The sequence was present, but the endpoint was not labeled. |
| `999` | No information | No usable information was available for the endpoint. |

---

# Main resource outputs

The principal release files are stored in:

```text
processed_data/processed_data/
```

| File | Description |
|---|---|
| `maomao_sequence_pivot.csv` | Main sequence-level MAOMAO resource. |
| `metadata.json` | Resource-level metadata, vocabulary, hierarchy, provenance summaries, processing rules, and statistics. |
| `maomao_ambiguous_support.csv` | Supporting evidence associated with ambiguous annotations. |
| `audit_endpoint_counts.csv` | Endpoint-level counts used to audit the final resource. |
| `audit_hierarchy_changes.csv` | Record of annotations modified or supported by hierarchy rules. |

## Main pivot structure

The main pivot contains one row per unique peptide sequence:

```csv
id,sequence,toxic,cytotoxic,hemolytic,cytolysis,neurotoxic,embryotoxic,ichthyotoxic
seq_1,ACDEFGHIK,1,1,1,999,999,999,999
seq_2,LLVLLAAAG,0,999,999,999,999,999,999
```

| Column | Description |
|---|---|
| `id` | MAOMAO sequence identifier. |
| `sequence` | Standardized peptide sequence. |
| Endpoint columns | Evidence code for each harmonized toxicity endpoint. |

The same `id` is preserved across the master resource, numerical representations, endpoint-specific datasets, and generated splits.

---

# Repository structure

The GitHub repository contains the source code, reproducible notebooks, configuration files, and Snakemake workflows required to construct and process MAOMAO.

```text
.
├── general_configs/
│   └── random_seeds_30.csv
│
├── notebooks_and_scripts/
│   ├── parsing_data/
│   ├── integrate_negative_evidence_of_datasets/
│   ├── integrate_organism_data/
│   ├── integrating_and_cleaning_data/
│   ├── pivoting_data_and_hierarchical_structure/
│   ├── preprocessing_for_split/
│   ├── dataset_caracterization/
│   └── sequence_distribution_analysis/
│
├── pipelines/
│   ├── data/
│   │   └── <endpoint>/
│   │       └── sequences.csv
│   ├── numerical_representations/
│   │   ├── config/config.yaml
│   │   ├── README.md
│   │   └── Snakefile
│   └── split_dataset/
│       ├── config/config.yaml
│       ├── README.md
│       └── Snakefile
│
├── src/
│   ├── maomao/
│   └── building_models/
│
├── LICENSE.txt
├── pyproject.toml
└── README.md
```

# Data availability

The GitHub repository contains the source code, notebooks, configuration files, and reproducible workflows required to construct and process MAOMAO.

The complete MAOMAO data release and associated computational artefacts are distributed through a single Zenodo record:

- **MAOMAO data release:** Zenodo DOI and link pending.

The archived release includes the harmonized peptide toxicity resource, source- and resource-level metadata, provenance records, audit tables, numerical sequence representations, and reproducible endpoint-specific dataset partitions.

Due to their size and data-distribution requirements, the following directories are not included directly in the GitHub repository:

```text
maomao/
├── raw_data/
│   └── <source>/
│
├── processed_data/
│   ├── toxic_effect_classification/
│   │   └── <source>/
│   ├── integrating_and_cleaning_data/
│   │   └── <endpoint>/
│   └── processed_data/
│       ├── maomao_sequence_pivot.csv
│       ├── maomao_ambiguous_support.csv
│       ├── audit_endpoint_counts.csv
│       ├── audit_hierarchy_changes.csv
│       └── metadata.json
│
├── numerical_representation_data/
│   └── maomao/
│       ├── sylphy_embedding/
│       │   └── <model_alias>/
│       └── sylphy_one_hot/
│           └── one_hot/
│
└── split_process/
    └── maomao_<endpoint>/
```

## Directory description

| Directory | Description |
|---|---|
| `raw_data/` | Original source files used to construct the resource. |
| `processed_data/toxic_effect_classification/` | Source-specific parsed and standardized datasets with source metadata. |
| `processed_data/integrating_and_cleaning_data/` | Endpoint-level integrated positive, negative, ambiguous, organism, and provenance outputs. |
| `processed_data/processed_data/` | Final MAOMAO master resource, metadata, and audit files. |
| `notebooks_and_scripts/` | Reproducible notebooks for parsing, integration, hierarchy construction, characterization, and split preparation. |
| `src/maomao/` | MAOMAO-specific reusable Python modules. |
| `src/building_models/` | Supporting utilities for numerical representations, preprocessing, and model-related workflows. |
| `numerical_representation_data/` | Protein language model embeddings and one-hot representations. |
| `pipelines/` | Configuration-driven Snakemake workflows. |
| `pipelines/data/` | Endpoint-specific binary datasets prepared for splitting. |
| `split_process/` | Reproducible train, validation, and test partitions organized by endpoint and representation. |
| `general_configs/` | Shared workflow configuration, including the predefined random seeds. |

---

# Software requirements

## Core software

| Software | Purpose |
|---|---|
| Python 3.11 or later | Resource processing and workflow implementation. |
| Snakemake 9.x | Workflow dependency management and reproducible execution. |
| Sylphy | Protein language model embeddings and one-hot sequence representations. |
| BioSieve | Reproducible dataset partitioning. |
| ROXY | Dataset characterization. |

A CUDA-capable GPU is recommended for large protein language model embeddings but is not required for the resource-construction notebooks or one-hot encoding.

## Python dependencies

Python dependencies are declared in:

```text
pyproject.toml
```

Important packages used throughout the repository include pandas, NumPy, PyYAML, scikit-learn, Jupyter, Snakemake, and model-specific dependencies required by Sylphy.

Some tokenizer-backed protein language models may also require packages such as `protobuf` or `sentencepiece`.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/kren-ai-lab/maomao.git
cd maomao
```

## 2. Create and activate a Conda environment

```bash
conda create -n maomao python=3.11
conda activate maomao
```

## 3. Install the repository package

```bash
python -m pip install -e .
```

## 4. Install Snakemake

```bash
conda install -c conda-forge snakemake
```

Ensure that the `sylphy` and `biosieve` command-line programs required by the enabled workflows are available in the active environment.

## 5. Verify the installation

```bash
python -c "import maomao; print('MAOMAO package available')"
python -m snakemake --version
sylphy --help
biosieve --help
```

A workflow can be checked without executing it by running a Snakemake dry run:

```bash
cd pipelines/numerical_representations
python -m snakemake -n -p
```

---

# Reconstructing MAOMAO

The resource is constructed in sequential, inspectable stages.

```text
Raw source files
       ↓
Source-specific parsing and sequence standardization
       ↓
Endpoint-specific evidence integration
       ↓
Organism and negative-evidence integration
       ↓
Terminology harmonization
       ↓
Ontology-guided hierarchy application
       ↓
Sequence-level pivot and metadata
       ↓
Numerical representations and benchmark-ready splits
```

## 1. Parse source datasets

Source-specific notebooks are located in:

```text
notebooks_and_scripts/parsing_data/
```

Each parser converts a source into standardized sequence and annotation files stored under:

```text
processed_data/toxic_effect_classification/<source>/
```

Typical source outputs include processed endpoint datasets, modified or corrected sequence datasets, detected invalid or unsupported sequences, and source-level `metadata.json` files.

## 2. Integrate endpoint evidence

Endpoint-specific integration notebooks are located in:

```text
notebooks_and_scripts/integrating_and_cleaning_data/
```

They produce harmonized endpoint directories under:

```text
processed_data/integrating_and_cleaning_data/<endpoint>/
```

Depending on available evidence, these directories can contain:

- `positive.csv`;
- `negative.csv`;
- `ambiguous_data.csv`;
- `sequence_negative_evidece.csv`;
- `sequence_by_organism.csv`;
- `metadata.json`.

## 3. Build the master resource

The master resource is generated with:

```text
notebooks_and_scripts/pivoting_data_and_hierarchical_structure/build_maomao_master.ipynb
```

This stage builds the sequence-level endpoint pivot, applies the ontology hierarchy, preserves direct ambiguity, assigns stable sequence identifiers, creates audit tables, compiles resource-level metadata, and writes the final outputs to `processed_data/processed_data/`.

---

# Computational workflows

The current repository includes independent Snakemake workflows for numerical representation generation and dataset splitting. Each workflow has its own `README.md`, `Snakefile`, and `config/config.yaml`.

## 1. Numerical representations

Location:

```text
pipelines/numerical_representations/
```

Purpose:

- generate one numerical representation per unique MAOMAO sequence;
- preserve MAOMAO identifiers and endpoint annotations;
- optionally analyze the representation space;
- provide reusable features for downstream datasets and splits.

### Supported representation families

The current repository structure includes:

| Family | Representation |
|---|---|
| ANKH | `ankh2_ext1`, `ankh3_large` |
| ESM-2 | `esm2_t6_8M_UR50D`, `esm2_t12_35M_UR50D`, `esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D` |
| ESM-C | `esmc_300m` |
| Mistral-Prot | `mistral_prot_v1_134M` |
| ProtTrans | `prot_bert`, `prot_t5_xl_uniref50` |
| Baseline | `one_hot` |

### Main outputs

```text
numerical_representation_data/maomao/
├── sylphy_embedding/
│   └── <model_alias>/
│       ├── embeddings.csv
│       └── full_data.csv
└── sylphy_one_hot/
    └── one_hot/
        ├── encoded.csv
        └── full_data.csv
```

`full_data.csv` preserves the original MAOMAO identifiers and endpoint columns alongside the generated numerical features.

Detailed workflow documentation is available in `pipelines/numerical_representations/README.md`.

## 2. Endpoint-specific split preparation

The master pivot contains multiple endpoint columns, whereas each supervised split requires one binary `label` column.

The preparation notebook is:

```text
notebooks_and_scripts/preprocessing_for_split/creating_raw_data.ipynb
```

For each endpoint, it preserves `id` and `sequence`, renames the selected endpoint to `label`, retains only explicit negative (`0`) and positive (`1`) records, and exports the binary dataset to:

```text
pipelines/data/<endpoint>/sequences.csv
```

Ambiguous (`2`), unlabeled (`3`), and no-information (`999`) records are not used as binary split labels.

## 3. Dataset splitting

Location:

```text
pipelines/split_dataset/
```

Purpose:

- combine endpoint-specific binary labels with reusable numerical representations;
- generate reproducible train, validation, and test partitions;
- evaluate multiple random seeds;
- validate generated folds;
- record invalid or infeasible partitions rather than silently accepting them.

### Supported strategies

The current workflow supports random K-fold, stratified K-fold, and distance-aware K-fold when enabled and configured.

### Typical split outputs

```text
split_process/
└── maomao_<endpoint>/
    └── <representation>_<source>/
        └── <strategy>/
            └── seed_<seed>/
                ├── fold_00/
                │   ├── train.csv
                │   ├── val.csv
                │   └── test.csv
                ├── fold_01/
                ├── params_split.yaml
                ├── kfold_report.json
                ├── split_summary.csv
                ├── biosieve_split.stdout.log
                ├── biosieve_split.stderr.log
                └── DONE.txt
```

Detailed workflow documentation is available in `pipelines/split_dataset/README.md`.

---

# Quick start

The following example uses the existing MAOMAO master resource and generates neurotoxicity splits.

## Step 1. Verify the master dataset

```bash
ls processed_data/processed_data/maomao_sequence_pivot.csv
```

## Step 2. Generate or reuse numerical representations

Edit `pipelines/numerical_representations/config/config.yaml` and run:

```bash
cd pipelines/numerical_representations

python -m snakemake \
    --cores 1 \
    --rerun-incomplete \
    --latency-wait 60 \
    -p
```

## Step 3. Prepare the endpoint-specific dataset

Run:

```text
notebooks_and_scripts/preprocessing_for_split/creating_raw_data.ipynb
```

Confirm that the endpoint file exists:

```bash
ls pipelines/data/neurotoxic/sequences.csv
```

## Step 4. Configure neurotoxicity splitting

Edit `pipelines/split_dataset/config/config.yaml`:

```yaml
dataset:
  name: "maomao_neurotoxic"
  representation_dataset: "maomao"
  input_data: "../data/neurotoxic/sequences.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"

output:
  root: "../../split_process"
  include_dataset_folder: true
  materialized_root: "../../split_process_inputs/neurotoxic"
```

## Step 5. Run a dry run

```bash
cd pipelines/split_dataset
python -m snakemake -n -p
```

## Step 6. Generate the splits

```bash
python -m snakemake \
    --cores 8 \
    --rerun-incomplete \
    --latency-wait 60 \
    -p
```

Generated outputs are written to:

```text
split_process/maomao_neurotoxic/
```

To process another endpoint, change the endpoint-specific dataset name, input path, and temporary materialization directory while continuing to reuse `representation_dataset: "maomao"`.

---

# Configuration

Each Snakemake workflow is configured independently.

| Configuration file | Purpose |
|---|---|
| `pipelines/numerical_representations/config/config.yaml` | Numerical representation generation and optional representation-space analysis. |
| `pipelines/split_dataset/config/config.yaml` | Source selection, representations, split strategies, validation, seeds, and output directories. |
| `general_configs/random_seeds_30.csv` | Shared set of 30 random seeds used for reproducible split generation. |

Relative paths are preferred because they make the workflows portable across local computers and computing clusters.

---

# Reproducibility and traceability

MAOMAO supports reproducibility through:

- source-level directories and `metadata.json` files;
- endpoint-level metadata and evidence integration outputs;
- stable sequence identifiers preserved across all workflow stages;
- explicit hierarchy audit tables;
- configuration-driven Snakemake workflows;
- predefined random seeds in `general_configs/random_seeds_30.csv`;
- split validation, reports, status files, and execution logs.

The same input files, software environment, configuration files, and seeds can be used to reproduce generated resource artefacts and partitions.

---

# Citation

When using MAOMAO, please cite the associated resource publication and archived software release.

## Software

```text
Zenodo citation pending.
```

## Manuscript

```text
MAOMAO: An Ontology-Guided FAIR Resource for Harmonized Peptide Toxicity Data.
Publication details pending.
```

## BibTeX

```bibtex

```

---

# License

This repository is distributed under the **MIT License**.

See [LICENSE.txt](LICENSE.txt) for the complete license text.

Individual source datasets may retain their original terms, licenses, and citation requirements. Consult the corresponding source-level metadata and original provider before redistributing or reusing source-specific files.

---

# Authors and contact

Developed by the MAOMAO contributors and the Kren AI Lab.

For questions regarding the resource or software, contact:

- David Medina-Ortiz: [david.medina@umag.cl](mailto:david.medina@umag.cl)
