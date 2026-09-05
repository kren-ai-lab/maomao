# Resource Use and Worked Examples

This directory contains a set of practical notebooks demonstrating how the released **MAOMAO** resource can be accessed, explored, and reused in downstream computational analyses.

These notebooks are intended as **usage examples for external users**. They operate on the released MAOMAO assets and are conceptually separate from the notebooks used to construct, harmonize, and validate the resource.

The examples cover:

- direct access to the sequence-level MAOMAO resource;
- descriptive exploration of endpoint annotations;
- reuse of released numerical representations;
- reconstruction of released benchmark partitions;
- a minimal neurotoxicity classification example;
- an illustrative quantitative activity regression example;
- exploration of sequence-level MAOMAO cards.

The goal is to demonstrate how MAOMAO can be integrated into common exploratory data analysis and machine-learning workflows without requiring users to reconstruct the resource from the original source datasets.

---

# Table of contents

- [Notebook overview](#notebook-overview)
- [Required MAOMAO assets](#required-maomao-assets)
- [Running the examples](#running-the-examples)
- [Notebook descriptions](#notebook-descriptions)
  - [00 — Getting started](#00--getting-started)
  - [01 — Resource characterization](#01--resource-characterization)
  - [02 — Neurotoxicity classification](#02--neurotoxicity-classification)
  - [03 — Quantitative activity regression](#03--quantitative-activity-regression)
  - [04 — Sequence-card exploration](#04--sequence-card-exploration)
- [Generated outputs](#generated-outputs)
- [Scope and interpretation](#scope-and-interpretation)
- [Reproducibility](#reproducibility)

---

# Notebook overview

The notebooks are organized as a progressive set of examples:

```text
MAOMAO public release
        │
        ├── 00_getting_started.ipynb
        │      Access, inspect, query, and connect released assets
        │
        ├── 01_resource_characterization.ipynb
        │      Descriptive exploration of the released resource
        │
        ├── 02_neurotoxicity_classification.ipynb
        │      Released split + released representation → classifier
        │
        ├── 03_quantitative_activity_regression.ipynb
        │      Quantitative measurements + representation → regression
        │
        └── 04_sequence_card_exploration.ipynb
               Inspect and query sequence-level cards
```

| Notebook | Purpose | Main MAOMAO assets |
|---|---|---|
| `00_getting_started.ipynb` | Minimal entry point for navigating and querying the public release. | Core Layer; optionally Embedding and Benchmark Layers |
| `01_resource_characterization.ipynb` | Descriptive characterization of sequences, endpoint states, co-occurrence, class balance, and release coverage. | Core Layer; optionally Embedding and Benchmark Layers |
| `02_neurotoxicity_classification.ipynb` | Minimal supervised-learning example using an existing neurotoxicity split and a released PLM representation. | Core, Embedding, and Benchmark Layers |
| `03_quantitative_activity_regression.ipynb` | Illustrative regression example using released quantitative toxicity measurements. | Core and Embedding Layers |
| `04_sequence_card_exploration.ipynb` | Minimal example for loading, inspecting, and querying sequence cards. | `sequence_profiles/sequence_cards.jsonl.gz` |

The notebooks can be inspected independently, although new users are encouraged to begin with `00_getting_started.ipynb`.

---

# Required MAOMAO assets

Installation and complete data-restoration instructions are provided in the main repository `README.md`.

The notebooks support both the operational repository structure:

```text
maomao/
├── processed_data/
├── numerical_representation_data/
├── split_process/
├── sequence_profiles/
└── ...
```

and, where applicable, the conceptual directory structure distributed with the archived release:

```text
maomao_release/
├── core_layer/
├── embedding_layer/
├── benchmark_layer/
├── descriptor_layer/
└── documentation_layer/
```

The notebooks automatically search common MAOMAO locations. An explicit root can also be supplied through the environment:

```bash
export MAOMAO_ROOT=/path/to/maomao
```

or by changing:

```python
MAOMAO_ROOT_OVERRIDE = None
```

inside the corresponding notebook.

Generated example outputs are written outside the frozen MAOMAO release data under:

```text
results_how_to_use_maomao/
```

when applicable.

---

# Running the examples

First install MAOMAO following the instructions in the repository root:

```bash
conda activate maomao
python -m pip install -e .
```

Jupyter must also be available in the active environment.

From the directory containing these notebooks, start Jupyter using, for example:

```bash
jupyter lab
```

The recommended execution order is:

```text
00_getting_started.ipynb
        ↓
01_resource_characterization.ipynb
        ↓
02_neurotoxicity_classification.ipynb
        ↓
03_quantitative_activity_regression.ipynb
        ↓
04_sequence_card_exploration.ipynb
```

The order is pedagogical rather than computational: the notebooks do not depend on outputs produced by previous examples.

---

# Notebook descriptions

## 00 — Getting started

```text
00_getting_started.ipynb
```

This notebook is the minimal entry point for direct reuse of the MAOMAO public release.

It demonstrates how to:

- locate and validate released MAOMAO assets;
- load the sequence-level master resource;
- inspect stable sequence identifiers;
- inspect the endpoint and evidence-state model;
- access metadata and audit tables;
- query peptides using endpoint evidence states;
- retrieve one peptide and inspect associated supporting records;
- discover released numerical representations;
- discover endpoint-specific benchmark partitions;
- connect a released split to numerical features using the stable MAOMAO `id`;
- optionally export a user-defined subset.

A central reuse principle demonstrated in this notebook is the stable identifier contract:

```text
MAOMAO master resource
          id
           │
           ├── numerical representations
           ├── endpoint-specific datasets
           ├── benchmark partitions
           └── additional resource records
```

This allows users to move between MAOMAO components without matching sequences manually.

The notebook does not reconstruct MAOMAO or regenerate numerical representations or benchmark partitions.

---

## 01 — Resource characterization

```text
01_resource_characterization.ipynb
```

This notebook provides a descriptive characterization of the frozen sequence-level MAOMAO resource.

The analyses include:

- number of peptide entities and sequence-length distribution;
- evidence-state composition by toxicity endpoint;
- positive and negative records available for conventional binary reuse;
- endpoint-level class balance;
- positive endpoint co-occurrence;
- multi-endpoint positive annotation burden;
- sequence-length comparisons between explicit positive and negative records;
- hierarchy-consistency checks;
- coverage of released numerical representations;
- inventory of released benchmark partitions.

The five MAOMAO evidence states are retained throughout the descriptive analysis:

| Code | State |
|---:|---|
| `0` | Negative |
| `1` | Positive |
| `2` | Ambiguous |
| `3` | Unlabeled |
| `999` | No information |

For binary descriptive comparisons, only explicit `0` and `1` records are considered negative and positive classes. Other states remain part of the resource and are summarized separately rather than converted into binary labels.

The statistical comparisons in this notebook are exploratory and should not be interpreted as evidence of causal relationships between sequence properties and toxicity.

---

## 02 — Neurotoxicity classification

```text
02_neurotoxicity_classification.ipynb
```

This notebook demonstrates immediate supervised reuse of MAOMAO using:

1. an existing released neurotoxicity train/validation/test partition; and
2. an existing released protein language model representation.

The workflow follows:

```text
released neurotoxicity split
        id + label
             +
released PLM representation
        id + features
             ↓
          join on id
             ↓
   conventional classifier
```

The example compares three standard models with fixed settings:

- Logistic Regression;
- linear Support Vector Machine;
- Random Forest.

Only explicit binary neurotoxicity labels are used:

```text
0 = negative
1 = positive
```

The notebook reports conventional classification metrics, including accuracy, balanced accuracy, precision, recall, F1 score, Matthews correlation coefficient, ROC AUC, and average precision, together with per-class results and confusion matrices.

No hyperparameter optimization is performed and no new benchmark partition is generated.

The example is intended to demonstrate **interoperability and immediate computational reuse**, not to establish a new state-of-the-art neurotoxicity predictor or replace the frozen MAOMAO benchmark design.

---

## 03 — Quantitative activity regression

```text
03_quantitative_activity_regression.ipynb
```

This notebook demonstrates how quantitative toxicity measurements retained during MAOMAO processing can be reused for a simple regression workflow.

The current example searches for standardized datasets associated with:

- `HC50`;
- `LC50`;
- `LD50`;
- `MHC`.

Because these measurements represent different experimental endpoints and may use different units, they are not pooled into a single numerical target.

Instead, the notebook:

1. discovers available quantitative datasets;
2. summarizes endpoint and unit coverage;
3. selects one homogeneous endpoint–unit combination with sufficient observations;
4. maps exact peptide sequences to stable MAOMAO identifiers;
5. aggregates repeated measurements for the same exact sequence using the median;
6. connects the selected sequences to one released PLM representation;
7. applies a `log10` transformation to strictly positive activity values;
8. creates a reproducible demonstration train/test partition;
9. compares a median baseline with a Ridge regression model;
10. reports MAE, RMSE, R², and Spearman correlation.

The workflow illustrates how MAOMAO can support analyses beyond categorical endpoint classification while preserving the experimental endpoint and measurement unit.

This example is not intended as a validated quantitative toxicity prediction benchmark.

---

## 04 — Sequence-card exploration

```text
04_sequence_card_exploration.ipynb
```

This notebook provides a minimal example for working with the released MAOMAO sequence-card collection:

```text
sequence_profiles/sequence_cards.jsonl.gz
```

It demonstrates how to:

- open the compressed JSON Lines collection;
- inspect one complete sequence card;
- retrieve a specific card using its stable MAOMAO `id`;
- summarize final endpoint states;
- inspect direct-source evidence counts;
- inspect ontology-supported annotations;
- inspect toxicity targets supported by direct positive evidence;
- perform a simple endpoint/status query across the collection.

For example, cards can be filtered to retrieve peptide entities with:

```python
ENDPOINT = "neurotoxic"
STATUS = "positive"
```

The sequence cards complement the compact sequence-level pivot by providing a more detailed, human-readable representation of evidence associated with individual peptide entities.

This notebook is intentionally lightweight and does not generate additional datasets by default.

---

# Generated outputs

Where analytical outputs are generated, they are stored under:

```text
results_how_to_use_maomao/
```

Typical structure:

```text
results_how_to_use_maomao/
├── 01_resource_characterization/
│   ├── figures/
│   └── tables/
│
├── 02_neurotoxicity_classification/
│   ├── figures/
│   └── tables/
│
└── 03_quantitative_activity_regression/
    ├── figures/
    └── tables/
```

Outputs include descriptive summary tables, model metrics, predictions, analysis metadata, and publication-quality PNG/PDF figures where applicable.

`00_getting_started.ipynb` writes a user-defined subset only when optional export is explicitly enabled.

`04_sequence_card_exploration.ipynb` is an interactive inspection example and does not create output files by default.

The released MAOMAO files themselves should be treated as read-only. User-generated derivatives and example results are therefore written separately from the distributed resource.

---

# Scope and interpretation

These notebooks are designed to answer the practical question:

> **How can an external user immediately reuse MAOMAO after downloading the resource?**

They therefore emphasize resource navigation, interoperability, exploratory analysis, and minimal downstream examples.

They do **not**:

- reconstruct MAOMAO from the original sources;
- redefine endpoint annotations;
- overwrite ambiguous or unlabeled states;
- regenerate the ontology;
- replace released benchmark partitions;
- perform model selection for a new MAOMAO reference predictor;
- claim that the illustrative classification or regression models represent state-of-the-art toxicity prediction.

Resource-construction workflows remain documented separately in the main repository.

---

# Reproducibility

The examples preserve MAOMAO's reproducibility principles by:

- reading frozen released resource files;
- preserving stable MAOMAO sequence identifiers;
- reusing released train/validation/test membership when available;
- using released numerical representations rather than silently regenerating features;
- explicitly recording model parameters and analysis settings;
- separating generated demonstration outputs from the public release;
- using deterministic random states where new illustrative sampling is required.

For the complete description of MAOMAO construction, data layers, installation, configuration, data availability, and citation, see the main repository `README.md`.
