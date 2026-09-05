# Changelog

All notable release-level changes to MAOMAO are recorded in this file.

## [0.1.0] — 2026-07-27 — Initial release candidate

### Added

- Initial ontology-guided and provenance-aware MAOMAO resource.
- Integration metadata for 54 peptide-toxicity sources.
- Final sequence-level pivot containing 71,701 unique peptide sequences and 7 controlled toxicity endpoints.
- Complete sequence–endpoint cross-product containing 501,907 combinations.
- Mutually exclusive positive, negative, ambiguous, unlabeled, and no-information evidence states.
- Positive-only endpoint hierarchy with ambiguity blocking and sequence-level hierarchy audit records.
- Ambiguity-support records for 13,027 unique peptide sequences.
- Descriptor Layer containing 41 sequence-derived descriptors.
- Embedding Layer containing ten protein language model representations and one one-hot baseline.
- Binary Benchmark Layer for `toxic`, `cytotoxic`, `hemolytic`, and `neurotoxic`.
- Two partitioning strategies, 30 random seeds, 2,640 configurations, and 13,200 train–validation–test fold instances.
- Centralized Documentation Layer with vocabulary, schema, licensing, versioning, and inventory records.

### Release status

This entry describes the package prepared for deposit on Zenodo for maomao v0.1.0 in July 2026.


## [1.0.0] — 2026-09-02

### Changed

- Replaced the sequential `seq_<integer>` identifiers with stable sequence-derived identifiers using the format `sha256_<64_lowercase_hex>`.
- Generated each identifier from the UTF-8 encoding of the normalized uppercase peptide sequence after whitespace removal.
- Propagated the new identifiers to the Core, Descriptor, Embedding, One-hot, and Benchmark Layers.
- Added the sequence-derived identifier column to `maomao_ambiguous_support.csv` and `audit_hierarchy_changes.csv`.
- Updated resource metadata, inventories, file dimensions, and checksums.
- Constrained pandas to `>=3.0.1,<4.0` for compatibility with Sylphy 0.2.0.

### Compatibility

- This is an identifier-breaking release. Files from v0.1.0 cannot be joined directly to v1.0.0 using `id`.
- The peptide sequences, toxicity endpoint states, descriptor values, numerical representations, and benchmark fold memberships remain scientifically unchanged.


## [1.1.0] — 2026-09-04

### Added

* Added `anti_mammalian_cells` as an eighth controlled toxicity endpoint.
* Extended the toxicity ontology so that `anti_mammalian_cells` is represented as a child of `cytotoxic`, alongside `hemolytic` and `cytolysis`.
* Added harmonized quantitative toxicity measurements to the Core Layer, including HC50, LC50, LD50, and MHC records with measurement relations, values, units, experimental context, and source provenance.
* Added sequence cards to the Core Layer, providing one compact machine-readable profile per normalized peptide sequence.
* Added sequence-level activity evidence and source-level evidence tables supporting the sequence cards.
* Added selected physicochemical descriptors to the sequence cards while retaining the complete 41-descriptor matrix in the Descriptor Layer as the canonical descriptor resource.
* Added a JSON Schema, metadata, checksums, JSON and HTML examples, and export utilities for sequence-card inspection and reuse.
* Added practical usage notebooks demonstrating resource access, descriptive characterization, neurotoxicity classification, quantitative activity regression, and sequence-card exploration.

### Changed

* Updated the master sequence-level resource and resource metadata to represent eight final toxicity endpoints.
* Updated hierarchy propagation so that positive `anti_mammalian_cells` evidence supports `cytotoxic` and, transitively, `toxic`.
* Updated endpoint summaries, audit information, and metadata to reflect the expanded controlled vocabulary.
* Expanded the Core Layer to include standardized quantitative toxicity measurements and sequence-level cards alongside the harmonized sequence resource and supporting evidence records.
* Expanded resource documentation to describe quantitative toxicity measurements, sequence-card interpretation, and downstream reuse examples.
