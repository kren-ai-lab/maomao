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
