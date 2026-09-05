# MAOMAO: An Ontology-Guided FAIR Resource for Harmonized Peptide Toxicity Data

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Snakemake](https://img.shields.io/badge/Snakemake-9.x-green.svg)](https://snakemake.github.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22312156-blue?style=flat-square)](https://doi.org/10.5281/zenodo.22312156)

Nicole Soto-García<sup>1</sup>, Julián García-Vinuesa<sup>1</sup>, Roberto Uribe-Paredes<sup>1</sup>, Leandro Murgas-Saavedra<sup>1</sup>, Karen Oróstica<sup>2</sup>, Jorge González-Puelma<sup>3,4</sup>, Marcelo Navarrete<sup>3,4</sup>, Frederic Cadet<sup>5</sup>, and David Medina-Ortiz<sup>1,*</sup>.<br>

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
- [Data availability](#data-availability)
  - [Restoring the project data structure](#restoring-the-project-data-structure)
  - [Directory description](#directory-description)
- [Software requirements](#software-requirements)
- [Installation](#installation)
- [Reconstructing MAOMAO](#reconstructing-maomao)
- [Computational workflows](#computational-workflows)
  - [Numerical representations](#1-numerical-representations)
  - [Endpoint-specific split preparation](#2-endpoint-specific-split-preparation)
  - [Dataset splitting](#3-dataset-splitting)
- [Resource use examples](#resource-use-examples)
- [Quick start](#quick-start)
- [Configuration](#configuration)
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
- distinguishes positive, negative, ambiguous, unlabeled, and no-information evidence states;
- preserves source-level provenance and endpoint-specific metadata;
- records hierarchy-derived annotation changes;
- produces a sequence-level pivot resource;
- integrates standardized quantitative toxicity measurements;
- generates compact sequence cards for direct sequence-level inspection;
- generates numerical representations for computational reuse;
- creates reproducible endpoint-specific dataset partitions;
- provides practical notebooks demonstrating downstream resource reuse.

The final resource is organized around stable sequence identifiers, explicit endpoint columns, structured metadata, audit tables, quantitative toxicity measurements, sequence-level profiles, and reusable workflow outputs.

# Resource scope

MAOMAO currently organizes evidence for the following final toxicity endpoints:

| Endpoint | Description |
|---|---|
| `toxic` | General evidence that a peptide is toxic. |
| `cytotoxic` | Toxicity affecting cells. |
| `hemolytic` | Lysis or damage of red blood cells. |
| `cytolysis` | Evidence associated with cell lysis. |
| `anti_mammalian_cells` | Toxicity affecting mammalian cells. |
| `neurotoxic` | Toxicity affecting the nervous system. |
| `embryotoxic` | Toxicity affecting embryos or embryonic development. |
| `ichthyotoxic` | Toxicity affecting fish. |

Source-specific terminology is normalized during processing so that heterogeneous annotations can be integrated into this common endpoint vocabulary.

# Ontology and evidence model

## Endpoint hierarchy

MAOMAO uses the following toxicity hierarchy:

```text
Toxic
├── Cytotoxic
│   ├── Hemolytic
│   ├── Cytolysis
│   └── Anti-mammalian cells
├── Neurotoxic
├── Embryotoxic
└── Ichthyotoxic
```

Positive evidence can propagate upward through this hierarchy. For example, positive `anti_mammalian_cells` evidence supports `cytotoxic` and, transitively, `toxic`.

Negative, ambiguous, unlabeled, and no-information states are not propagated through the hierarchy. Direct ambiguity at a parent endpoint is retained and is not overwritten by positive hierarchical support.

## Evidence encoding

The sequence-level pivot uses the following codes:

| Code | Evidence state | Meaning |
|---:|---|---|
| `0` | Negative | Explicit negative evidence for the endpoint. |
| `1` | Positive | Explicit or hierarchy-supported positive evidence. |
| `2` | Ambiguous | Conflicting or unresolved evidence. |
| `3` | Unlabeled | The sequence was present, but the endpoint was not labeled. |
| `999` | No information | No usable information was available for the endpoint. |

`999` denotes absence of usable endpoint information and must not be interpreted as negative evidence.

# Main resource outputs

The principal release files are stored in:

```text
processed_data/processed_data/
```

| File | Description |
|---|---|
| `maomao_sequence_pivot.csv` | Main sequence-level MAOMAO resource. |
| `maomao_toxicity_measurements.csv` | Standardized quantitative toxicity measurements linked to MAOMAO sequences. |
| `maomao_sequence_pivot_with_toxicity_properties.csv` | Sequence-level pivot augmented with available quantitative toxicity properties. |
| `metadata.json` | Resource-level metadata, vocabulary, hierarchy, provenance summaries, processing rules, and statistics. |
| `metadata_toxicity_properties.json` | Metadata describing the quantitative toxicity-property integration. |
| `maomao_ambiguous_support.csv` | Supporting evidence associated with ambiguous annotations. |
| `audit_endpoint_counts.csv` | Endpoint-level counts used to audit the final resource. |
| `audit_hierarchy_changes.csv` | Record of annotations modified or supported by hierarchy rules. |
| `audit_toxicity_property_cross_reference.csv` | Audit linking standardized toxicity measurements to MAOMAO sequences. |
| `audit_toxicity_property_unmatched_sequences.csv` | Audit of quantitative-property records that could not be linked to the master sequence resource. |

## Main pivot structure

The main pivot contains one row per unique peptide sequence:

```csv
id,sequence,toxic,cytotoxic,hemolytic,cytolysis,neurotoxic,embryotoxic,ichthyotoxic,anti_mammalian_cells
sha256_c729ebc224388368ab8c8df88487ef137ad8bd5097651cf67c37bda5622c9f9a,ACDEFGHIK,1,1,1,999,999,999,999,999
sha256_a816c180c9e987c35d04962bb7db2b17827c950dcf36706d5d611e2163171a4d,LLVLLAAAG,0,999,999,999,999,999,999,0
```

| Column | Description |
|---|---|
| `id` | Stable sequence-derived identifier generated as `sha256_<digest>` from the normalized peptide sequence. |
| `sequence` | Standardized peptide sequence. |
| Endpoint columns | Evidence code for each harmonized toxicity endpoint. |

The same `id` is preserved across the master resource, quantitative toxicity measurements, sequence profiles, numerical representations, endpoint-specific datasets, and generated splits.

## Quantitative toxicity properties

MAOMAO v1.1.0 retains standardized source-reported quantitative toxicity measurements when they can be linked to a normalized sequence. The current integrated measurement types are `HC50`, `LC50`, `LD50`, and `MHC`.

Where available, records preserve the measurement type, comparison relation, numerical value, reported error, unit, original reported value, experimental context, and source provenance. Measurement types and units remain distinct; MAOMAO does not silently pool them or assume automatic conversion between mass and molar concentration units.

## Sequence profiles

Compact sequence cards are distributed as part of the **Core Layer** in the Zenodo release under `core_layer/sequence_profiles/`. They do not constitute a separate conceptual layer.

Each card combines final endpoint states, direct evidence counts, ontology support, negative-evidence provenance, toxicity-target categories, quantitative toxicity measurements, and selected physicochemical descriptors for one normalized peptide sequence.

The card collection includes `sequence_cards.jsonl.gz`, detailed sequence- and source-level evidence tables, a JSON Schema, metadata, checksums, and JSON/HTML examples. The complete 41-descriptor matrix remains in the **Descriptor Layer**; only a compact descriptor subset is shown in each card.

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
│   ├── integrate_toxicity_properties/
│   ├── integrating_and_cleaning_data/
│   ├── pivoting_data_and_hierarchical_structure/
│   ├── preprocessing_for_split/
│   ├── dataset_caracterization/
│   ├── sequence_distribution_analysis/
│   ├── sequence_cards/
│   └── how_to_use_maomao/
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
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
└── README.md
```

# Data availability

The GitHub repository contains the source code, notebooks, configuration files, and reproducible workflows required to construct and process MAOMAO.

The complete MAOMAO data release and associated computational artefacts are distributed through a single Zenodo record:

- **MAOMAO data release (version 1.1.0):** [https://doi.org/10.5281/zenodo.22312156](https://doi.org/10.5281/zenodo.22312156)

The archived release is organized into distribution layers:

```text
maomao_release/
├── benchmark_layer/
├── core_layer/
│   ├── processed_data/
│   ├── raw_data/
│   ├── README.md
│   └── sequence_profiles/
├── descriptor_layer/
├── documentation_layer/
│   └── results_how_to_use_maomao/
├── embedding_layer/
└── README.md
```

The **Core Layer** contains the harmonized peptide toxicity resource, source- and resource-level metadata, provenance and audit records, standardized quantitative toxicity measurements, and sequence profiles. The **Descriptor Layer** contains sequence-derived descriptors. The **Embedding Layer** contains protein language model and one-hot numerical representations. The **Benchmark Layer** contains reproducible endpoint-specific train/validation/test partitions. The **Documentation Layer** contains release documentation and the derived outputs from the practical MAOMAO usage examples.

The layer directories are Zenodo distribution containers and are not the same as the operational directory structure of the GitHub repository.

Due to their size and data-distribution requirements, the following operational data directories are not included directly in the GitHub repository:

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
│       ├── maomao_sequence_pivot_with_toxicity_properties.csv
│       ├── maomao_toxicity_measurements.csv
│       ├── maomao_ambiguous_support.csv
│       ├── audit_endpoint_counts.csv
│       ├── audit_hierarchy_changes.csv
│       ├── audit_toxicity_property_cross_reference.csv
│       ├── audit_toxicity_property_unmatched_sequences.csv
│       ├── metadata.json
│       └── metadata_toxicity_properties.json
│
├── sequence_profiles/
│   ├── sequence_cards.jsonl.gz
│   ├── sequence_activity_evidence.csv.gz
│   ├── sequence_source_evidence.csv.gz
│   ├── sequence_card_schema.json
│   ├── metadata.json
│   ├── README.md
│   ├── CHECKSUMS.sha256
│   └── examples/
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

## Restoring the project data structure

The Zenodo layer directories organize the public release but do not correspond directly to the operational paths used by the repository workflows.

The following instructions restore the Core Layer contents to their expected locations within the cloned MAOMAO repository.

### 1. Verify the working directory

Run the following commands from the root of the cloned `maomao` repository:

```bash
test -f pyproject.toml
test -d pipelines
```

If either command fails, navigate to the repository root before continuing.

### 2. Download the Core Layer

Create a temporary directory for the downloaded MAOMAO release:

```bash
mkdir -p downloads/maomao_release
```

Download `core_layer.zip` from Zenodo:

```bash
curl -fL   "https://zenodo.org/api/records/22312156/files/core_layer.zip/content"   -o downloads/core_layer.zip
```

### 3. Extract the Core Layer

Extract the downloaded archive into the temporary release directory:

```bash
unzip downloads/core_layer.zip -d downloads/maomao_release
```

After extraction, the files have the following intermediate structure:

```text
downloads/
├── core_layer.zip
└── maomao_release/
    └── core_layer/
        ├── processed_data/
        ├── raw_data/
        ├── README.md
        └── sequence_profiles/
```

This is the Zenodo distribution structure. The data directories must still be copied into the operational repository locations.

### 4. Restore the operational directories

Create the directories expected by the repository and usage notebooks:

```bash
mkdir -p raw_data processed_data sequence_profiles
```

Copy the Core Layer contents to their corresponding operational locations:

```bash
cp -R downloads/maomao_release/core_layer/raw_data/. raw_data/
cp -R downloads/maomao_release/core_layer/processed_data/. processed_data/
cp -R downloads/maomao_release/core_layer/sequence_profiles/. sequence_profiles/
```

The trailing `/.` copies the contents of each directory without retaining `core_layer` as an additional directory level.

### 5. Verify the restored data

Confirm that the principal sequence-level resource is available:

```bash
test -f processed_data/processed_data/maomao_sequence_pivot.csv
```

Confirm that the quantitative toxicity measurements are available:

```bash
test -f processed_data/processed_data/maomao_toxicity_measurements.csv
```

Confirm that the sequence-card collection is available:

```bash
test -f sequence_profiles/sequence_cards.jsonl.gz
```

The restored Core Layer should produce the following operational data structure alongside the cloned repository:

```text
maomao/
├── raw_data/
├── processed_data/
├── sequence_profiles/
├── pipelines/
├── pyproject.toml
└── README.md
```

> **Note:** `downloads/maomao_release/` is a temporary staging location and is not used by the workflows after the data have been copied. The Embedding Layer contributes `numerical_representation_data/`, while the Benchmark Layer contributes `split_process/`. The Descriptor and Documentation Layers can remain in their Zenodo distribution structure when used for consultation. In particular, `documentation_layer/results_how_to_use_maomao/` contains derived outputs from the usage notebooks and is not an operational source directory of the GitHub repository.

## Directory description

| Directory | Description |
|---|---|
| `raw_data/` | Original source files used to construct the resource. |
| `processed_data/toxic_effect_classification/` | Source-specific parsed and standardized datasets with source metadata. |
| `processed_data/integrating_and_cleaning_data/` | Endpoint-level integrated positive, negative, ambiguous, organism, and provenance outputs. |
| `processed_data/processed_data/` | Final MAOMAO master resource, quantitative toxicity measurements, metadata, and audit files. |
| `sequence_profiles/` | Sequence-card data restored from the Zenodo Core Layer for direct inspection and reuse. |
| `notebooks_and_scripts/` | Reproducible notebooks for parsing, integration, hierarchy construction, toxicity-property integration, characterization, sequence-card generation, split preparation, and practical reuse examples. |
| `src/maomao/` | MAOMAO-specific reusable Python modules. |
| `src/building_models/` | Supporting utilities for numerical representations, preprocessing, and model-related workflows. |
| `numerical_representation_data/` | Protein language model embeddings and one-hot representations restored from the Embedding Layer. |
| `pipelines/` | Configuration-driven Snakemake workflows. |
| `pipelines/data/` | Endpoint-specific binary datasets prepared for splitting. |
| `split_process/` | Reproducible train, validation, and test partitions restored from the Benchmark Layer. |
| `general_configs/` | Shared workflow configuration, including the predefined random seeds. |

# Software requirements

## Core software

| Software | Purpose |
|---|---|
| Python 3.11 or later | Resource processing and workflow implementation. |
| Snakemake 9.x | Workflow dependency management and reproducible execution. |
| Sylphy | Protein language model embeddings and one-hot sequence representations. |
| BioSieve | Reproducible dataset partitioning. |
| ROXY (pinned legacy API) | Dataset characterization. |

A CUDA-capable GPU is recommended for large protein language model embeddings. Resource-construction notebooks and one-hot encoding can be executed without GPU acceleration; however, platform-specific dependency issues may occur on macOS.

> **macOS compatibility note:** The one-hot workflow may encounter `OMP: Error #15` in some macOS environments when multiple OpenMP runtimes are loaded by dependencies such as PyTorch and scikit-learn. If this occurs, a temporary workaround is to set:
>
> ```bash
> export KMP_DUPLICATE_LIB_OK=TRUE
> ```
>
> This issue is specific to some macOS environments and does not affect the recommended Linux workflow.

## Python dependencies

Python dependencies are declared in:

```text
pyproject.toml
```

Runtime dependencies are declared in `pyproject.toml`. Important packages include pandas, NumPy, PyYAML, scikit-learn, ROXY, and the model-specific dependencies provided by Sylphy.

ROXY is installed from a pinned pre-refactor commit because the MAOMAO dataset-characterization utilities depend on the legacy `roxy.eda.summary` and `roxy.report` APIs, which are not available in current ROXY releases.

Snakemake is installed separately through Conda, as described below. Jupyter is only required to execute the notebooks interactively and must be installed separately if needed.

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
conda install -c conda-forge -c bioconda snakemake
```

Ensure that the `sylphy` and `biosieve` command-line programs required by the enabled workflows are available in the active environment.

## 5. Verify the installation

Verify that MAOMAO and the required command-line programs are available in the active environment:

```bash
python -c "import maomao; print('MAOMAO package available')"
python -m snakemake --version
sylphy --version
biosieve --version
python -c "from roxy.eda.summary import build_report; from roxy.report import dataset_report_to_html; print('ROXY legacy API available')"
```

Before running a Snakemake dry run, restore the Core Layer as described in [Restoring the project data structure](#restoring-the-project-data-structure). From the repository root, confirm that the numerical-representation workflow input is available:

```bash
test -f processed_data/processed_data/maomao_sequence_pivot.csv
```

If this command fails, restore the Core Layer before continuing. Once the required file is available, check the workflow without executing it:

```bash
cd pipelines/numerical_representations
python -m snakemake -n -p
```

---

# Reconstructing MAOMAO

The resource is constructed through sequential and inspectable stages.

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
Quantitative toxicity-property integration
       ↓
Sequence-card generation
       ↓
Numerical representation generation
       ↓
Endpoint-specific dataset preparation
       ↓
Reproducible benchmark partitioning
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

## 4. Integrate quantitative toxicity properties

Quantitative HC50, LC50, LD50, and MHC records are standardized and linked to the MAOMAO sequence identifiers through:

```text
notebooks_and_scripts/integrate_toxicity_properties/integrating_toxicity_properties.ipynb
```

The resulting measurement, augmented pivot, metadata, and audit files are written to `processed_data/processed_data/`.

## 5. Generate sequence cards

Sequence-card construction is demonstrated in:

```text
notebooks_and_scripts/sequence_cards/build_sequence_cards.ipynb
```

Reusable card-generation utilities are implemented under:

```text
src/maomao/sequence_cards/
```

The distributed card collection is part of the Core Layer in Zenodo. Individual cards can also be exported as readable JSON and HTML using `notebooks_and_scripts/sequence_cards/export_sequence_card.py`.

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

* generate reproducible train, validation, and test partitions from endpoint-specific binary datasets;
* generate representation-independent random and stratified partitions;
* generate representation-specific distance-aware partitions when enabled;
* evaluate multiple random seeds;
* validate generated folds;
* retain compact split files containing only sequence identifiers and labels;
* record invalid or infeasible partitions rather than silently accepting them.

### Supported strategies

The current workflow supports random K-fold, stratified K-fold, and distance-aware K-fold when enabled and configured.

Random and stratified partitions are generated once per dataset variant and can be reused with any compatible numerical representation. Distance-aware partitions remain associated with the numerical representation used to calculate distances.

### Typical split outputs

For random and stratified strategies:

```text
split_process/
└── maomao_<endpoint>/
    └── no_reduced
        └── <random_kfold|stratified_kfold>/
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

Each permanent `train.csv`, `val.csv`, and `test.csv` contains the sequence identifier and binary label. Numerical features can be reconstructed by joining these files with the corresponding representation-level `full_data.csv` using the sequence identifier.

Detailed workflow documentation is available in `pipelines/split_dataset/README.md`.


### Reconstructing Representation-Specific Benchmark Files

The permanent `train.csv`, `val.csv`, and `test.csv` files contain only the sequence identifier and binary label:

```text
id,label
```

Numerical features are not stored directly in each split file. Instead, representation-specific datasets must be reconstructed by joining the selected split with the corresponding `full_data.csv` file generated by the numerical-representation workflow in the embedding layer.

This design avoids duplicating large numerical matrices across strategies, seeds, folds, and endpoint-specific datasets. The same random or stratified partition can therefore be reused with any compatible numerical representation while preserving the original train, validation, and test assignments.

Representation files are stored under:

```text
numerical_representation_data/maomao/
├── sylphy_embedding/
│   └── <model_alias>/
│       └── full_data.csv
└── sylphy_one_hot/
    └── one_hot/
        └── full_data.csv
```

For example, the following code reconstructs a subset using the `ankh2_ext1` representation:

```python
import pandas as pd

split_df = pd.read_csv(
    "split_process/maomao_cytotoxic/"
    "no_reduced/stratified_kfold/"
    "seed_113/fold_00/train.csv"
)

representation_df = pd.read_csv(
    "numerical_representation_data/maomao/"
    "sylphy_embedding/ankh2_ext1/full_data.csv"
).drop(columns=["label"], errors="ignore")

reconstructed_df = split_df.merge(
    representation_df,
    on="id",
    how="inner",
)

reconstructed_df.to_csv(
    "reconstructed_train.csv",
    index=False,
)
```

The same procedure can be applied to `val.csv` and `test.csv`, as well as to any of the available numerical representations.

The binary label stored in the benchmark split should be preserved as the target variable. Any existing `label` column in the representation file should be removed before the merge to avoid duplicated or conflicting target columns.

The `id` column must remain unchanged across the endpoint-specific dataset, benchmark partitions, and numerical-representation files. After reconstruction, each row contains the fixed benchmark assignment, the endpoint-specific binary label, and the numerical features required for downstream model training and evaluation.


---

# Resource use examples

Practical notebooks demonstrating direct reuse of the released MAOMAO resource are available under:

```text
notebooks_and_scripts/how_to_use_maomao/
```

The examples cover:

- locating and loading released MAOMAO assets;
- descriptive resource characterization;
- reuse of released numerical representations and benchmark partitions;
- a minimal neurotoxicity classification example;
- a quantitative toxicity regression example;
- sequence-card exploration.

These notebooks are usage demonstrations rather than new MAOMAO benchmark models. Their derived tables and figures are archived in the Zenodo Documentation Layer under:

```text
documentation_layer/results_how_to_use_maomao/
```

See `notebooks_and_scripts/how_to_use_maomao/README.md` for details.

---

# Quick start

The following example uses the existing MAOMAO master resource and generates neurotoxicity splits.

## Step 1. Verify the master dataset

```bash
ls processed_data/processed_data/maomao_sequence_pivot.csv
```

## Step 2. Generate or reuse numerical representations when required

Numerical representations are required for reconstructing representation-specific benchmark files. Random and stratified partitions can be generated directly from the endpoint-specific binary dataset.

To generate or reuse numerical representations, edit `pipelines/numerical_representations/config/config.yaml` and run:

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

When using MAOMAO, please cite the associated resource publication and archived data release.

## Data release

```text
Soto-Garcia, N., García-Vinuesa, J., Uribe-Paredes, R., Murgas, L., Oróstica, K., González-Puelma, J., Navarrete, M., Cadet, F., & Medina-Ortiz, D. (2026). MAOMAO: An Ontology-Guided FAIR Resource for Harmonized Peptide Toxicity Data (Version 1.1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.22312156
```

### Data release BibTeX

```bibtex
@dataset{soto2026maomao22312156,
  author       = {Soto-Garcia, Nicole and
                  Garc{\'i}a-Vinuesa, Juli{\'a}n and
                  Uribe-Paredes, Roberto and
                  Murgas, Leandro and
                  Oróstica, Karen and
                  González-Puelma, Jorge and
                  Navarrete, Marcelo and
                  CADET, Frederic and
                  Medina-Ortiz, David},
  title        = {MAOMAO: An Ontology-Guided FAIR Resource for Harmonized Peptide Toxicity Data},
  month        = sep,
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.1.0},
  doi          = {10.5281/zenodo.22312156},
  url          = {https://doi.org/10.5281/zenodo.22312156}
}
```

## Preprint

```text
Soto-Garcia, N., Uribe-Paredes, R., Murgas, L., Oróstica, K., González-Puelma, J., Navarrete, M., Cadet, F., & Medina-Ortiz, D. (2026). MAOMAO: An Ontology-Guided FAIR Resource for Harmonized Peptide Toxicity Data. bioRxiv. https://doi.org/10.64898/2026.07.29.741655
```

### Preprint BibTeX

```bibtex
@article{soto2026maomao,
  title     = {MAOMAO: An Ontology-Guided FAIR Resource for Harmonized Peptide Toxicity Data},
  author    = {Soto-Garcia, Nicole and
               Uribe-Paredes, Roberto and
               Murgas, Leandro and
               Or{\'o}stica, Karen and
               Gonz{\'a}lez-Puelma, Jorge and
               Navarrete, Marcelo and
               Cadet, Frederic and
               Medina-Ortiz, David},
  journal   = {bioRxiv},
  year      = {2026},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.64898/2026.07.29.741655},
  url       = {https://doi.org/10.64898/2026.07.29.741655}
}
```

---

# License

This repository is distributed under the **MIT License**.

See [LICENSE](LICENSE) for the complete license text.

Individual source datasets may retain their original terms, licenses, and citation requirements. Consult the corresponding source-level metadata and original provider before redistributing or reusing source-specific files.

---

# Authors and contact

Developed by the MAOMAO contributors and the Kren AI Lab.

For questions regarding the resource or software, contact:

- David Medina-Ortiz: [david.medina@umag.cl](mailto:david.medina@umag.cl)
