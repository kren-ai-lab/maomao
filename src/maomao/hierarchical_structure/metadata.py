
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


FINAL_ENDPOINTS = [
    "toxic",
    "cytotoxic",
    "hemolytic",
    "cytolysis",
    "neurotoxic",
    "embryotoxic",
    "ichthyotoxic",
    "anti_mammalian_cells"
]

LABEL_ENCODING = {
    "0": "negative: direct negative evidence for the endpoint",
    "1": (
        "positive: direct positive evidence or positive evidence propagated "
        "through the hierarchy when the parent endpoint is not ambiguous"
    ),
    "2": (
        "ambiguous: conflicting or otherwise ambiguous evidence for the "
        "specific endpoint; hierarchical positivity does not overwrite it"
    ),
    "3": (
        "unlabeled: sequence retained in an endpoint-specific source without "
        "an explicit positive or negative assignment"
    ),
    "999": "no information: no annotation is available for the endpoint",
}

CODE_TO_STATUS = {
    0: "negative",
    1: "positive",
    2: "ambiguous",
    3: "unlabeled",
    999: "no_information",
}

MISSING_METADATA_VALUES = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "no information",
    "not available",
    "unknown",
}


def _snake_case(value):
    value = str(value).strip()
    value = re.sub(r"[^0-9A-Za-z]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()


SOURCE_KEY_ALIASES = {
    "type_source": "source_type",
    "static_dynamic": "resource_behavior",
    "year_of_publication": "publication_year",
    "last_update_date": "last_update_date",
    "download_date": "download_date",
    "file_format": "file_format",
    "peptide_property": "peptide_property",
    "dataset_information": "annotation_content",
    "unit_of_measurement": "unit_of_measurement",
    "obtaining_negative_dataset": "negative_dataset_origin",
    "repository_or_server": "repository_or_server",
}


def _normalize_metadata_keys(value, aliases=None):
    aliases = aliases or {}

    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            normalized_key = aliases.get(
                _snake_case(key),
                _snake_case(key),
            )
            normalized[normalized_key] = _normalize_metadata_keys(
                item,
                aliases=aliases,
            )
        return normalized

    if isinstance(value, list):
        return [
            _normalize_metadata_keys(item, aliases=aliases)
            for item in value
        ]

    return value


def _to_builtin(value):
    if isinstance(value, dict):
        return {
            str(key): _to_builtin(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    if (
        not isinstance(value, (dict, list, tuple, set))
        and pd.isna(value)
    ):
        return None

    return value


def _is_available_metadata_value(value):
    if value is None:
        return False

    if isinstance(value, str):
        return (
            value.strip().lower()
            not in MISSING_METADATA_VALUES
        )

    return True


def _load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _relative_path(path, anchor):
    try:
        return str(
            path.resolve().relative_to(anchor.resolve())
        )
    except ValueError:
        return str(path.resolve())


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_field_coverage(
    entries,
    metadata_key="metadata",
):
    fields = sorted(
        {
            field
            for entry in entries
            for field in entry.get(
                metadata_key,
                {},
            )
        }
    )
    total = len(entries)

    coverage = {}
    for field in fields:
        available = sum(
            _is_available_metadata_value(
                entry.get(
                    metadata_key,
                    {},
                ).get(field)
            )
            for entry in entries
        )
        coverage[field] = {
            "available": int(available),
            "missing": int(total - available),
            "total": int(total),
            "coverage_percent": (
                round(
                    100 * available / total,
                    2,
                )
                if total
                else 0.0
            ),
        }

    return coverage


def collect_source_metadata(
    source_metadata_root,
    project_root,
):
    entries = []
    errors = []

    if not source_metadata_root.exists():
        return {
            "metadata_root": _relative_path(
                source_metadata_root,
                project_root,
            ),
            "n_sources": 0,
            "sources": [],
            "field_coverage": {},
            "read_errors": [
                {
                    "path": str(
                        source_metadata_root
                    ),
                    "error": (
                        "directory_not_found"
                    ),
                }
            ],
        }

    for metadata_path in sorted(
        source_metadata_root.glob(
            "*/metadata.json"
        )
    ):
        source_name = metadata_path.parent.name

        try:
            raw_metadata = _load_json(
                metadata_path
            )
            normalized_metadata = (
                _normalize_metadata_keys(
                    raw_metadata,
                    aliases=SOURCE_KEY_ALIASES,
                )
            )
            entries.append(
                {
                    "source_name": source_name,
                    "metadata_file": (
                        _relative_path(
                            metadata_path,
                            project_root,
                        )
                    ),
                    "metadata": (
                        normalized_metadata
                    ),
                }
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(
                {
                    "source_name": source_name,
                    "metadata_file": (
                        _relative_path(
                            metadata_path,
                            project_root,
                        )
                    ),
                    "error": str(exc),
                }
            )

    return {
        "metadata_root": _relative_path(
            source_metadata_root,
            project_root,
        ),
        "source_identifier_policy": (
            "The source directory name is retained "
            "as the provenance identifier. Source "
            "names appearing in endpoint integration "
            "metadata are preserved as generated by "
            "that stage and are not silently "
            "reconciled."
        ),
        "key_normalization": (
            "Metadata keys are converted to "
            "lowercase snake_case; source values are "
            "retained without semantic modification."
        ),
        "n_sources": int(len(entries)),
        "source_names": [
            entry["source_name"]
            for entry in entries
        ],
        "field_coverage": (
            _metadata_field_coverage(entries)
        ),
        "sources": entries,
        "read_errors": errors,
    }


def collect_endpoint_metadata(
    endpoint_metadata_root,
    project_root,
):
    entries = []
    errors = []

    if not endpoint_metadata_root.exists():
        return {
            "metadata_root": _relative_path(
                endpoint_metadata_root,
                project_root,
            ),
            "n_endpoint_directories": 0,
            "endpoints": [],
            "read_errors": [
                {
                    "path": str(
                        endpoint_metadata_root
                    ),
                    "error": (
                        "directory_not_found"
                    ),
                }
            ],
        }

    for metadata_path in sorted(
        endpoint_metadata_root.glob(
            "*/metadata.json"
        )
    ):
        directory_endpoint = (
            metadata_path.parent.name
        )

        try:
            raw_metadata = _load_json(
                metadata_path
            )
            normalized_metadata = (
                _normalize_metadata_keys(
                    raw_metadata
                )
            )
            entries.append(
                {
                    "endpoint_directory": (
                        directory_endpoint
                    ),
                    "canonical_endpoint": (
                        "cytolysis"
                        if directory_endpoint
                        == "cytolytic"
                        else directory_endpoint
                    ),
                    "metadata_file": (
                        _relative_path(
                            metadata_path,
                            project_root,
                        )
                    ),
                    "metadata": (
                        normalized_metadata
                    ),
                }
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(
                {
                    "endpoint_directory": (
                        directory_endpoint
                    ),
                    "metadata_file": (
                        _relative_path(
                            metadata_path,
                            project_root,
                        )
                    ),
                    "error": str(exc),
                }
            )

    return {
        "metadata_root": _relative_path(
            endpoint_metadata_root,
            project_root,
        ),
        "n_endpoint_directories": int(
            len(entries)
        ),
        "canonical_term_mapping": {
            "cytolytic": "cytolysis",
        },
        "endpoints": entries,
        "read_errors": errors,
    }


def reconstruct_status_before_hierarchy(
    evidence_long,
    hierarchy_audit,
):
    before = evidence_long[
        ["sequence", "endpoint", "status"]
    ].rename(
        columns={
            "status": (
                "status_before_hierarchy"
            )
        }
    )

    if (
        hierarchy_audit is None
        or hierarchy_audit.empty
        or "status_before_hierarchy"
        not in hierarchy_audit
    ):
        return before

    audit_before = hierarchy_audit[
        [
            "sequence",
            "endpoint",
            "status_before_hierarchy",
        ]
    ].drop_duplicates(
        ["sequence", "endpoint"]
    )

    before = before.drop(
        columns="status_before_hierarchy"
    ).merge(
        audit_before,
        on=["sequence", "endpoint"],
        how="left",
    )

    final_lookup = evidence_long.set_index(
        ["sequence", "endpoint"]
    )["status"]

    missing = before[
        "status_before_hierarchy"
    ].isna()

    before.loc[
        missing,
        "status_before_hierarchy",
    ] = [
        final_lookup.loc[
            (sequence, endpoint)
        ]
        for sequence, endpoint in before.loc[
            missing,
            ["sequence", "endpoint"],
        ].itertuples(
            index=False,
            name=None,
        )
    ]

    return before


def status_summary(
    long_table,
    status_column,
):
    statuses = [
        "positive",
        "negative",
        "ambiguous",
        "unlabeled",
        "no_information",
    ]

    summary = (
        long_table
        .groupby(
            ["endpoint", status_column]
        )
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=statuses,
            fill_value=0,
        )
        .astype(int)
    )

    records = []
    for endpoint, row in summary.iterrows():
        record = {
            "endpoint": endpoint,
            "n_sequence_endpoint_rows": int(
                row.sum()
            ),
            "n_annotated": int(
                row.sum()
                - row["no_information"]
            ),
        }
        record.update(
            {
                status: int(row[status])
                for status in statuses
            }
        )
        record["label_distribution"] = {
            str(code): int(row[status])
            for code, status in (
                CODE_TO_STATUS.items()
            )
        }
        records.append(record)

    return records


def final_pivot_summary(wide):
    records = []

    for endpoint in FINAL_ENDPOINTS:
        counts = (
            wide[endpoint]
            .astype("int64")
            .value_counts()
            .to_dict()
        )

        records.append(
            {
                "endpoint": endpoint,
                "n_sequence_endpoint_rows": int(
                    wide.shape[0]
                ),
                "n_annotated": int(
                    wide.shape[0]
                    - counts.get(999, 0)
                ),
                "positive": int(
                    counts.get(1, 0)
                ),
                "negative": int(
                    counts.get(0, 0)
                ),
                "ambiguous": int(
                    counts.get(2, 0)
                ),
                "unlabeled": int(
                    counts.get(3, 0)
                ),
                "no_information": int(
                    counts.get(999, 0)
                ),
                "label_distribution": {
                    str(code): int(
                        counts.get(code, 0)
                    )
                    for code in (
                        CODE_TO_STATUS
                    )
                },
            }
        )

    return records


def ambiguous_support_summary(
    wide,
    ambiguous_support,
):
    output = []

    for endpoint in FINAL_ENDPOINTS:
        n_ambiguous = int(
            wide[endpoint].eq(2).sum()
        )
        support_categories = {}

        if (
            ambiguous_support is not None
            and endpoint
            in ambiguous_support.columns
        ):
            values = (
                ambiguous_support[endpoint]
                .astype("string")
                .replace(
                    {
                        "<NA>": pd.NA,
                        "nan": pd.NA,
                        "999": pd.NA,
                    }
                )
                .dropna()
            )
            support_categories = {
                str(category): int(count)
                for category, count in (
                    values.value_counts()
                    .sort_index()
                    .items()
                )
            }

        output.append(
            {
                "endpoint": endpoint,
                "n_ambiguous": n_ambiguous,
                "positive_support_categories": (
                    support_categories
                ),
            }
        )

    return output


def hierarchy_statistics(
    hierarchy_audit,
):
    if (
        hierarchy_audit is None
        or hierarchy_audit.empty
    ):
        return {
            (
                "n_rows_with_hierarchical_"
                "positive_support"
            ): 0,
            "n_hierarchy_inferred": 0,
            (
                "n_hierarchy_blocked_by_"
                "ambiguity"
            ): 0,
            "n_hierarchical_conflicts": 0,
            "by_target_endpoint": [],
        }

    boolean_columns = [
        "is_hierarchy_inferred",
        "hierarchy_blocked_by_ambiguity",
        "has_hierarchical_conflict",
    ]
    audit = hierarchy_audit.copy()

    for column in boolean_columns:
        if column not in audit:
            audit[column] = False
        audit[column] = (
            audit[column]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )

    grouped = (
        audit
        .groupby("endpoint")
        .agg(
            n_rows_with_hierarchical_positive_support=(
                "endpoint",
                "size",
            ),
            n_hierarchy_inferred=(
                "is_hierarchy_inferred",
                "sum",
            ),
            n_hierarchy_blocked_by_ambiguity=(
                "hierarchy_blocked_by_ambiguity",
                "sum",
            ),
            n_hierarchical_conflicts=(
                "has_hierarchical_conflict",
                "sum",
            ),
        )
        .reset_index()
    )

    return {
        (
            "n_rows_with_hierarchical_"
            "positive_support"
        ): int(len(audit)),
        "n_hierarchy_inferred": int(
            audit[
                "is_hierarchy_inferred"
            ].sum()
        ),
        (
            "n_hierarchy_blocked_by_"
            "ambiguity"
        ): int(
            audit[
                (
                    "hierarchy_blocked_by_"
                    "ambiguity"
                )
            ].sum()
        ),
        "n_hierarchical_conflicts": int(
            audit[
                "has_hierarchical_conflict"
            ].sum()
        ),
        "by_target_endpoint": (
            grouped.to_dict(
                orient="records"
            )
        ),
    }


def sequence_statistics(wide):
    lengths = (
        wide["sequence"]
        .astype("string")
        .str.len()
    )

    return {
        "n_unique_sequences": int(
            wide["sequence"].nunique()
        ),
        "sequence_column": "sequence",
        "length_distribution": {
            "minimum": int(lengths.min()),
            "maximum": int(lengths.max()),
            "mean": round(
                float(lengths.mean()),
                4,
            ),
            "median": float(
                lengths.median()
            ),
        },
    }


def dataframe_audit_summary(
    dataframe,
    name,
):
    if dataframe is None:
        return {
            "name": name,
            "n_rows": 0,
            "n_columns": 0,
        }

    return {
        "name": name,
        "n_rows": int(
            dataframe.shape[0]
        ),
        "n_columns": int(
            dataframe.shape[1]
        ),
        "columns": list(
            dataframe.columns
        ),
    }


def output_file_metadata(output_root):
    descriptions = {
        "maomao_sequence_pivot.csv": (
            "Sequence-level pivot containing "
            "one mutually exclusive annotation "
            "code per toxicity endpoint."
        ),
        "maomao_ambiguous_support.csv": (
            "Endpoint-specific positive-support "
            "categories for ambiguous "
            "sequence-endpoint annotations."
        ),
        "audit_endpoint_counts.csv": (
            "Mutually exclusive endpoint counts "
            "calculated from the final pivot "
            "encoding."
        ),
        "audit_hierarchy_changes.csv": (
            "Traceability table for hierarchical "
            "positive support, applied inferences, "
            "conflicts, and cases blocked by "
            "parent ambiguity."
        ),
    }

    outputs = []

    for filename, description in (
        descriptions.items()
    ):
        path = output_root / filename
        record = {
            "filename": filename,
            "description": description,
            "exists": path.exists(),
        }

        if path.exists():
            record.update(
                {
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                    "sha256": _sha256(
                        path
                    ),
                }
            )

            try:
                table = pd.read_csv(path)
                record["n_rows"] = int(
                    table.shape[0]
                )
                record["n_columns"] = int(
                    table.shape[1]
                )
                record["columns"] = list(
                    table.columns
                )
            except Exception as exc:
                record[
                    "table_inspection_error"
                ] = str(exc)

        outputs.append(record)

    return outputs


def build_maomao_metadata(
    cfg,
    results,
    source_metadata_root,
    resource_version,
    repository_url,
    license_name,
):
    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    project_root = (
        cfg.input_root.parent.parent
    )

    wide = results["wide"]
    evidence_long = results[
        "evidence_long"
    ]
    hierarchy_audit = results[
        "hierarchy_audit"
    ]
    ambiguous_support = results[
        "ambiguous_support"
    ]

    before = (
        reconstruct_status_before_hierarchy(
            evidence_long,
            hierarchy_audit,
        )
    )

    source_metadata = (
        collect_source_metadata(
            source_metadata_root,
            project_root,
        )
    )
    endpoint_metadata = (
        collect_endpoint_metadata(
            cfg.input_root,
            project_root,
        )
    )

    assertions = results.get(
        "assertions"
    )
    assertions_records = (
        assertions.to_dict(
            orient="records"
        )
        if isinstance(
            assertions,
            pd.DataFrame,
        )
        else []
    )
    assertions_passed = (
        bool(
            assertions["passed"].all()
        )
        if isinstance(
            assertions,
            pd.DataFrame,
        )
        and "passed" in assertions
        else None
    )

    metadata = {
        "resource": {
            "name": "MAOMAO",
            "expanded_name": (
                "Metadata-Aware Ontology for "
                "Multi-source Annotation "
                "Organization"
            ),
            "title": (
                "MAOMAO: An Ontology-Guided "
                "FAIR Resource for Harmonized "
                "Peptide Toxicity Data"
            ),
            "version": resource_version,
            "generated_at": generated_at,
            "description": (
                "Ontology-guided, "
                "provenance-aware, and "
                "uncertainty-aware resource "
                "integrating multi-source "
                "peptide toxicity annotations."
            ),
            "repository": repository_url,
            "license": license_name,
            "resource_type": (
                "harmonized biological "
                "data resource"
            ),
        },
        "dataset": {
            "name": (
                "maomao_sequence_pivot"
            ),
            "filename": (
                "maomao_sequence_pivot.csv"
            ),
            "identifier_column": "id",
            "identifier_format": "sha256_<64_lowercase_hex>",
            "identifier_algorithm": "SHA-256",
            "identifier_input": (
                "UTF-8 encoding of the normalized uppercase "
                "peptide sequence with whitespace removed"
            ),
            "sequence_column": "sequence",
            "annotation_columns": (
                FINAL_ENDPOINTS
            ),
            "n_unique_sequences": int(
                wide.shape[0]
            ),
            "n_endpoints": len(
                FINAL_ENDPOINTS
            ),
            (
                "n_sequence_endpoint_rows"
            ): int(
                wide.shape[0]
                * len(FINAL_ENDPOINTS)
            ),
            "label_encoding": (
                LABEL_ENCODING
            ),
            "state_priority": (
                "endpoint-specific ambiguity "
                "is retained and cannot be "
                "overwritten by hierarchical "
                "positivity"
            ),
            "negative_evidence_policy": (
                "negative annotations are "
                "endpoint-specific and are "
                "never propagated through "
                "the hierarchy"
            ),
            "missing_information_policy": (
                "code 999 denotes absence of "
                "endpoint annotation and is "
                "distinct from negative and "
                "unlabeled evidence"
            ),
        },
        "processing_configuration": {
            "input_root": _relative_path(
                cfg.input_root,
                project_root,
            ),
            "source_metadata_root": (
                _relative_path(
                    source_metadata_root,
                    project_root,
                )
            ),
            "output_root": _relative_path(
                cfg.output_root,
                project_root,
            ),
            (
                "include_full_sequence_"
                "endpoint_cross_product"
            ): bool(
                cfg.include_full_cross_product
            ),
            "sequence_filters": {
                "canonical_residues": {
                    "applied": True,
                    "allowed_residues": str(
                        cfg.canonical_residues
                    ),
                },
                "length_filter": {
                    "applied": True,
                    "minimum": int(
                        cfg.min_length
                    ),
                    "maximum": int(
                        cfg.max_length
                    ),
                },
            },
        },
        (
            "controlled_vocabulary_and_"
            "hierarchy"
        ): {
            "version": "1.0",
            "description": (
                "Controlled vocabulary and "
                "hierarchy used to harmonize "
                "peptide toxicity annotations "
                "across heterogeneous sources."
            ),
            "hierarchy_policy": {
                "positive_propagation": True,
                "negative_propagation": False,
                "ambiguous_propagation": (
                    False
                ),
                "unlabeled_propagation": (
                    False
                ),
                (
                    "ambiguity_overrides_"
                    "hierarchy"
                ): True,
                (
                    "positive_support_is_"
                    "transitive"
                ): True,
            },
            "terms": [
                {
                    "term": "toxic",
                    "label_type": (
                        "broad toxicity "
                        "endpoint"
                    ),
                    "definition": (
                        "General evidence of "
                        "peptide toxicity."
                    ),
                },
                {
                    "term": "cytotoxic",
                    "label_type": (
                        "intermediate toxicity "
                        "endpoint"
                    ),
                    "definition": (
                        "Evidence of toxicity "
                        "affecting cells or "
                        "cell viability."
                    ),
                    "parent_terms": [
                        "toxic"
                    ],
                },
                {
                    "term": "anti_mammalian_cells",
                    "label_type": (
                        "specific toxicity "
                        "endpoint"
                    ),
                    "definition": (
                        "Evidence of peptide toxicity "
                        "affecting mammalian cells."
                    ),
                    "parent_terms": [
                        "cytotoxic",
                        "toxic",
                    ],
                },
                {
                    "term": "hemolytic",
                    "label_type": (
                        "specific toxicity "
                        "endpoint"
                    ),
                    "definition": (
                        "Evidence of erythrocyte "
                        "lysis or hemolytic "
                        "activity."
                    ),
                    "parent_terms": [
                        "cytotoxic",
                        "toxic",
                    ],
                },
                {
                    "term": "cytolysis",
                    "label_type": (
                        "specific toxicity "
                        "endpoint"
                    ),
                    "definition": (
                        "Evidence of cell lysis "
                        "or membrane-disruptive "
                        "toxicity."
                    ),
                    "parent_terms": [
                        "cytotoxic",
                        "toxic",
                    ],
                },
                {
                    "term": "cytolytic",
                    "label_type": (
                        "source-specific term"
                    ),
                    "definition": (
                        "Source-specific "
                        "cytolytic annotation "
                        "harmonized with "
                        "cytolysis."
                    ),
                    "harmonized_to": (
                        "cytolysis"
                    ),
                },
                {
                    "term": "neurotoxic",
                    "label_type": (
                        "specific toxicity "
                        "endpoint"
                    ),
                    "definition": (
                        "Evidence of toxicity "
                        "affecting the nervous "
                        "system."
                    ),
                    "parent_terms": [
                        "toxic"
                    ],
                },
                {
                    "term": "embryotoxic",
                    "label_type": (
                        "specific toxicity "
                        "endpoint"
                    ),
                    "definition": (
                        "Evidence of toxicity "
                        "affecting embryos or "
                        "embryonic development."
                    ),
                    "parent_terms": [
                        "toxic"
                    ],
                },
                {
                    "term": "ichthyotoxic",
                    "label_type": (
                        "specific toxicity "
                        "endpoint"
                    ),
                    "definition": (
                        "Evidence of toxicity "
                        "affecting fish."
                    ),
                    "parent_terms": [
                        "toxic"
                    ],
                },
            ],
        },
        "hierarchy_application": {
            "rules_applied": [
                {
                    "rule_id": "R1",
                    "description": (
                        "Positive Cytotoxic, "
                        "Neurotoxic, "
                        "Embryotoxic, or "
                        "Ichthyotoxic support "
                        "is propagated to "
                        "Toxic."
                    ),
                    "source_terms": [
                        "cytotoxic",
                        "neurotoxic",
                        "embryotoxic",
                        "ichthyotoxic",
                    ],
                    "target_term": "toxic",
                    "condition": (
                        "at least one child "
                        "endpoint has positive "
                        "support and Toxic is "
                        "not directly ambiguous"
                    ),
                    "action": (
                        "set Toxic to positive"
                    ),
                    "blocked_when": (
                        "Toxic has "
                        "endpoint-specific "
                        "ambiguous evidence"
                    ),
                    "propagation_policy": (
                        "positive_only"
                    ),
                },
                {
                    "rule_id": "R2",
                    "description": (
                        "Positive Hemolytic, Cytolysis, or "
                        "Anti-mammalian-cells support is "
                        "propagated to Cytotoxic."
                    ),
                    "source_terms": [
                        "hemolytic",
                        "cytolysis",
                        "anti_mammalian_cells",
                    ],
                    "target_term": (
                        "cytotoxic"
                    ),
                    "condition": (
                        "at least one child "
                        "endpoint has positive "
                        "support and Cytotoxic "
                        "is not directly "
                        "ambiguous"
                    ),
                    "action": (
                        "set Cytotoxic to "
                        "positive"
                    ),
                    "blocked_when": (
                        "Cytotoxic has "
                        "endpoint-specific "
                        "ambiguous evidence"
                    ),
                    "propagation_policy": (
                        "positive_only"
                    ),
                },
                {
                    "rule_id": "R3",
                    "description": (
                        "The source-specific "
                        "terms Cytolysis and "
                        "Cytolytic are "
                        "harmonized into "
                        "Cytolysis before final "
                        "state resolution."
                    ),
                    "source_terms": [
                        "cytolysis",
                        "cytolytic",
                    ],
                    "target_term": (
                        "cytolysis"
                    ),
                    "relation": (
                        "synonym_or_equivalent_"
                        "mapping"
                    ),
                    "action": (
                        "aggregate source "
                        "evidence and resolve "
                        "positive, negative, "
                        "ambiguous, unlabeled, "
                        "or no-information "
                        "status"
                    ),
                },
            ],
            "application_statistics": (
                hierarchy_statistics(
                    hierarchy_audit
                )
            ),
        },
        "input_provenance": {
            "source_level_metadata": (
                source_metadata
            ),
            (
                "endpoint_integration_"
                "metadata"
            ): endpoint_metadata,
        },
        "annotation_before_hierarchy": {
            "n_unique_sequences": int(
                wide.shape[0]
            ),
            "n_tasks": int(
                before[
                    "endpoint"
                ].nunique()
            ),
            "tasks": sorted(
                before[
                    "endpoint"
                ].unique().tolist()
            ),
            "task_summary": (
                status_summary(
                    before,
                    (
                        "status_before_"
                        "hierarchy"
                    ),
                )
            ),
        },
        "annotation_after_hierarchy": {
            "n_unique_sequences": int(
                wide.shape[0]
            ),
            "n_tasks": len(
                FINAL_ENDPOINTS
            ),
            "tasks": FINAL_ENDPOINTS,
            "task_summary": (
                final_pivot_summary(
                    wide
                )
            ),
        },
        "ambiguity": {
            "definition": (
                "Ambiguous annotations "
                "represent conflicting or "
                "otherwise unresolved "
                "evidence for the same "
                "sequence-endpoint pair."
            ),
            "hierarchy_policy": (
                "An ambiguous parent remains "
                "ambiguous even when a child "
                "endpoint supplies positive "
                "support."
            ),
            "support_file": (
                "maomao_ambiguous_support.csv"
            ),
            "endpoint_summary": (
                ambiguous_support_summary(
                    wide,
                    ambiguous_support,
                )
            ),
        },
        "sequence_statistics": (
            sequence_statistics(wide)
        ),
        "output_files": (
            output_file_metadata(
                cfg.output_root
            )
        ),
        "metadata_record": {
            "filename": (
                "maomao_metadata.json"
            ),
            "schema_version": "1.0",
            "self_checksum_included": (
                False
            ),
            (
                "self_checksum_exclusion_"
                "reason"
            ): (
                "A file cannot contain a "
                "stable checksum of its own "
                "complete contents."
            ),
        },
    }

    return _to_builtin(metadata)


def write_maomao_metadata(
    metadata,
    output_path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return output_path
