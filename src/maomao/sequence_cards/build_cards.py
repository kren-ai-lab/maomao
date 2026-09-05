from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import gzip
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd


STATUS_TO_CODE = {
    "negative": 0,
    "positive": 1,
    "ambiguous": 2,
    "unlabeled": 3,
    "no_information": 999,
}

CARD_SCHEMA_VERSION = "1.0.0"
SELECTED_DESCRIPTOR_COLUMNS = (
    "net_charge_pH",
    "boman_index",
    "fcr",
    "aa_entropy",
)
DESCRIPTOR_DECIMALS = 6


@dataclass(frozen=True)
class CardConfig:
    """Paths and release metadata used to build the sequence-card layer."""

    repo_root: Path
    output_dir: Path | None = None
    resource_version: str | None = None
    master_path: Path | None = None
    measurements_path: Path | None = None
    negative_evidence_metadata_path: Path | None = None
    toxicity_target_metadata_path: Path | None = None
    descriptors_path: Path | None = None
    numerical_representation_root: Path | None = None

    def resolved(self) -> "CardConfig":
        root = Path(self.repo_root).expanduser().resolve()
        core = root / "processed_data" / "processed_data"

        return CardConfig(
            repo_root=root,
            output_dir=(
                Path(self.output_dir).expanduser().resolve()
                if self.output_dir is not None
                else root / "sequence_profiles"
            ),
            resource_version=self.resource_version or _project_version(root),
            master_path=(
                Path(self.master_path).expanduser().resolve()
                if self.master_path is not None
                else core / "maomao_sequence_pivot.csv"
            ),
            measurements_path=(
                Path(self.measurements_path).expanduser().resolve()
                if self.measurements_path is not None
                else core / "maomao_toxicity_measurements.csv"
            ),
            negative_evidence_metadata_path=(
                Path(self.negative_evidence_metadata_path).expanduser().resolve()
                if self.negative_evidence_metadata_path is not None
                else root / "raw_data" / "evidence_negative_dataset.xlsx"
            ),
            toxicity_target_metadata_path=(
                Path(self.toxicity_target_metadata_path).expanduser().resolve()
                if self.toxicity_target_metadata_path is not None
                else root / "raw_data" / "tasks_by_source.xlsx"
            ),
            descriptors_path=(
                Path(self.descriptors_path).expanduser().resolve()
                if self.descriptors_path is not None
                else _first_existing(
                    [
                        root / "dataset_characterization" / "sequence_descriptors.csv",
                        root / "dataset_caracterization" / "sequence_descriptors.csv",
                        root / "processed_data" / "dataset_characterization" / "sequence_descriptors.csv",
                        root / "processed_data" / "dataset_caracterization" / "sequence_descriptors.csv",
                    ]
                )
            ),
            numerical_representation_root=(
                Path(self.numerical_representation_root).expanduser().resolve()
                if self.numerical_representation_root is not None
                else root / "numerical_representation_data" / "maomao"
            ),
        )


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def _project_version(repo_root: Path) -> str:
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return "development"

    try:
        import tomllib

        with pyproject.open("rb") as handle:
            return str(tomllib.load(handle).get("project", {}).get("version", "development"))
    except (OSError, ValueError):
        return "development"


def _load_hierarchy_module(repo_root: Path):
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from maomao.hierarchical_structure import define_hierarchy_and_structure as hierarchy

    return hierarchy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any, *, indent: int = 2) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
    os.replace(temporary, path)


def _native(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _load_master(cfg: CardConfig, hierarchy) -> tuple[pd.DataFrame, list[str]]:
    _require_file(cfg.master_path, "MAOMAO master pivot")
    master = pd.read_csv(cfg.master_path, low_memory=False)
    endpoints = list(hierarchy.ENDPOINTS)
    required = {"id", "sequence", *endpoints}
    missing = required.difference(master.columns)
    if missing:
        raise ValueError(f"Master pivot is missing columns: {sorted(missing)}")

    master = master.loc[:, ["id", "sequence", *endpoints]].copy()
    master["sequence"] = master["sequence"].map(hierarchy.normalize_sequence)
    expected_ids = master["sequence"].map(hierarchy.generate_sequence_id)

    if not master["id"].astype("string").equals(expected_ids.astype("string")):
        raise AssertionError("Master IDs do not match SHA-256 IDs derived from sequences.")
    if not master["id"].is_unique or not master["sequence"].is_unique:
        raise AssertionError("The master pivot must contain one row per ID and sequence.")

    for endpoint in endpoints:
        observed = set(pd.to_numeric(master[endpoint], errors="raise").astype(int).unique())
        invalid = observed.difference(STATUS_TO_CODE.values())
        if invalid:
            raise ValueError(f"{endpoint} contains invalid compact codes: {sorted(invalid)}")
        master[endpoint] = pd.to_numeric(master[endpoint], errors="raise").astype(int)

    return master, endpoints


def _rebuild_evidence(cfg: CardConfig, master: pd.DataFrame, endpoints: list[str], hierarchy):
    input_root = cfg.repo_root / "processed_data" / "integrating_and_cleaning_data"
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing integrated endpoint directory: {input_root}")

    hierarchy_cfg = hierarchy.Config(
        input_root=input_root,
        output_root=cfg.master_path.parent,
    )
    source_raw, row_raw, _, _ = hierarchy.collect_inputs(hierarchy_cfg)
    source_resolved, _ = hierarchy.resolve_source_evidence(source_raw)
    direct = hierarchy.build_direct_evidence(row_raw, source_resolved)
    complete = hierarchy.complete_grid(direct, hierarchy_cfg)
    evidence_long, hierarchy_audit = hierarchy.apply_positive_only_hierarchy(complete)

    evidence_long.insert(
        0,
        "id",
        evidence_long["sequence"].map(hierarchy.generate_sequence_id),
    )
    source_resolved.insert(
        0,
        "id",
        source_resolved["sequence"].map(hierarchy.generate_sequence_id),
    )

    expected_rows = len(master) * len(endpoints)
    if len(evidence_long) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows:,} sequence-endpoint evidence rows; found {len(evidence_long):,}."
        )

    evidence_ids = set(evidence_long["id"])
    master_ids = set(master["id"])
    if evidence_ids != master_ids:
        raise AssertionError(
            "Reconstructed evidence and the master pivot contain different sequence sets. "
            "Re-run build_maomao_master.ipynb before creating cards."
        )

    status_pivot = (
        evidence_long.assign(code=evidence_long["status"].map(STATUS_TO_CODE).astype(int))
        .pivot(index="id", columns="endpoint", values="code")
        .reindex(columns=endpoints)
    )
    master_codes = master.set_index("id")[endpoints].sort_index()
    status_pivot = status_pivot.sort_index()
    if not master_codes.equals(status_pivot.astype(master_codes.dtypes.to_dict())):
        differences = int((master_codes != status_pivot).sum().sum())
        raise AssertionError(
            f"The rebuilt evidence disagrees with the master pivot in {differences:,} cells. "
            "Do not generate cards from inconsistent layers."
        )

    return evidence_long, hierarchy_audit, source_resolved


def _read_annotation_matrices(
    cfg: CardConfig,
    hierarchy,
    master_ids: set[str],
    filename: str,
) -> tuple[dict[tuple[str, str], set[str]], list[str]]:
    values: dict[tuple[str, str], set[str]] = defaultdict(set)
    used_paths: list[str] = []
    input_root = cfg.repo_root / "processed_data" / "integrating_and_cleaning_data"

    for endpoint, folders in hierarchy.ENDPOINT_FOLDER_MAP.items():
        for folder in folders:
            path = input_root / folder / filename
            if not path.is_file():
                continue

            table = pd.read_csv(path, low_memory=False)
            if table.empty:
                used_paths.append(str(path.relative_to(cfg.repo_root)))
                continue

            if "sequence" in table.columns:
                sequence_column = "sequence"
            else:
                sequence_column = table.columns[0]

            table[sequence_column] = table[sequence_column].map(hierarchy.normalize_sequence)
            table = table[table[sequence_column].notna()].copy()
            table["id"] = table[sequence_column].map(hierarchy.generate_sequence_id)
            table = table[table["id"].isin(master_ids)]

            ignored = {sequence_column, "sequence", "id"}
            for category in [column for column in table.columns if column not in ignored]:
                active = pd.to_numeric(table[category], errors="coerce").eq(1)
                for identifier in table.loc[active, "id"].astype(str):
                    values[(identifier, endpoint)].add(str(category))

            used_paths.append(str(path.relative_to(cfg.repo_root)))

    return values, sorted(set(used_paths))


def _source_key(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _endpoint_key(value: Any) -> str:
    return "_".join(
        str(value)
        .strip()
        .casefold()
        .replace("-", " ")
        .split()
    )


def _load_negative_source_categories(
    cfg: CardConfig,
    endpoints: list[str],
) -> tuple[dict[tuple[str, str], set[str]], dict]:
    path = cfg.negative_evidence_metadata_path
    if path is None or not path.is_file():
        return {}, {
            "available": False,
            "path": str(path),
            "classified_source_endpoint_pairs": 0,
        }

    table = pd.read_excel(path)
    required = {"name source", "task", "negative dataset category"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "Negative-evidence metadata is missing columns: "
            f"{sorted(missing)}"
        )

    endpoint_set = set(endpoints)
    categories: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in table.loc[:, sorted(required)].itertuples(index=False, name=None):
        row_values = dict(zip(sorted(required), row))
        source = row_values["name source"]
        task = row_values["task"]
        category = row_values["negative dataset category"]
        if pd.isna(source) or pd.isna(task) or pd.isna(category):
            continue

        task_endpoints = {
            _endpoint_key(part)
            for part in str(task).replace(";", ",").split(",")
            if str(part).strip()
        }
        for endpoint in sorted(endpoint_set.intersection(task_endpoints)):
            categories[(_source_key(source), endpoint)].add(str(category).strip())

    return dict(categories), {
        "available": True,
        "path": str(path.relative_to(cfg.repo_root)),
        "classified_source_endpoint_pairs": len(categories),
    }


def _load_source_toxicity_targets(
    cfg: CardConfig,
    endpoints: list[str],
) -> tuple[dict[tuple[str, str], set[str]], dict]:
    path = cfg.toxicity_target_metadata_path
    if path is None or not path.is_file():
        return {}, {
            "available": False,
            "path": str(path),
            "classified_source_endpoint_pairs": 0,
        }

    table = pd.read_excel(path)
    required = {"name source", "task", "toxicity target"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "Toxicity-target metadata is missing columns: "
            f"{sorted(missing)}"
        )

    endpoint_set = set(endpoints)
    targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in table.loc[:, sorted(required)].itertuples(index=False, name=None):
        row_values = dict(zip(sorted(required), row))
        source = row_values["name source"]
        task = row_values["task"]
        target = row_values["toxicity target"]
        if pd.isna(source) or pd.isna(task) or pd.isna(target):
            continue

        task_endpoints = {
            _endpoint_key(part)
            for part in str(task).replace(";", ",").split(",")
            if str(part).strip()
        }
        for endpoint in sorted(endpoint_set.intersection(task_endpoints)):
            targets[(_source_key(source), endpoint)].add(str(target).strip())

    return dict(targets), {
        "available": True,
        "path": str(path.relative_to(cfg.repo_root)),
        "classified_source_endpoint_pairs": len(targets),
    }


def _load_measurements(cfg: CardConfig, master_ids: set[str]) -> tuple[dict[str, list[dict]], dict]:
    if not cfg.measurements_path.is_file():
        return {}, {
            "available": False,
            "path": str(cfg.measurements_path),
            "matched_measurements": 0,
        }

    table = pd.read_csv(cfg.measurements_path, low_memory=False)
    required = {"id", "measurement_type", "relation", "value", "unit"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Toxicity measurement table is missing columns: {sorted(missing)}")

    table = table[table["id"].isin(master_ids)].copy()
    if "in_maomao_pivot" in table.columns:
        table = table[table["in_maomao_pivot"].astype("boolean").fillna(False)]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in table.itertuples(index=False):
        record = {
            "type": str(row.measurement_type),
            "relation": str(row.relation),
            "value": _native(row.value),
            "unit": str(row.unit),
        }
        optional = {
            "error": getattr(row, "error", None),
            "reported_value": getattr(row, "raw_label", None),
            "source": getattr(row, "source", None),
            "source_dataset": getattr(row, "source_dataset", None),
        }
        for name, value in optional.items():
            native = _native(value)
            if native is not None:
                record[name] = native
        grouped[str(row.id)].append(record)

    for records in grouped.values():
        records.sort(key=lambda item: (item["type"], item["unit"], item["value"]))

    return dict(grouped), {
        "available": True,
        "path": str(cfg.measurements_path.relative_to(cfg.repo_root)),
        "matched_measurements": int(len(table)),
        "matched_sequences": int(table["id"].nunique()),
    }


def _read_id_set(path: Path, hierarchy) -> set[str]:
    header = pd.read_csv(path, nrows=0)
    if "id" in header.columns:
        return set(pd.read_csv(path, usecols=["id"], dtype="string")["id"].dropna().astype(str))
    if "sequence" in header.columns:
        sequences = pd.read_csv(path, usecols=["sequence"], dtype="string")["sequence"]
        return set(
            sequences.map(hierarchy.normalize_sequence)
            .dropna()
            .map(hierarchy.generate_sequence_id)
            .astype(str)
        )
    raise ValueError(f"Neither 'id' nor 'sequence' is present in {path}")


def _load_representations(cfg: CardConfig, hierarchy, endpoints: list[str]):
    descriptor_ids: set[str] = set()
    descriptor_summaries: dict[str, dict[str, float]] = {}
    descriptor_features: list[str] = []
    descriptor_path = cfg.descriptors_path
    if descriptor_path.is_file():
        descriptor_header = list(pd.read_csv(descriptor_path, nrows=0).columns)
        descriptor_features = [
            column
            for column in descriptor_header
            if column not in {"id", "sequence", *endpoints}
        ]
        missing_selected = set(SELECTED_DESCRIPTOR_COLUMNS).difference(descriptor_header)
        if missing_selected:
            raise ValueError(
                "Descriptor table is missing the columns selected for sequence cards: "
                f"{sorted(missing_selected)}"
            )

        identifier_columns = [
            column
            for column in ("id", "sequence")
            if column in descriptor_header
        ]
        descriptor_table = pd.read_csv(
            descriptor_path,
            usecols=[*identifier_columns, *SELECTED_DESCRIPTOR_COLUMNS],
            low_memory=False,
        )
        if "id" not in descriptor_table.columns:
            descriptor_table["sequence"] = descriptor_table["sequence"].map(
                hierarchy.normalize_sequence
            )
            descriptor_table["id"] = descriptor_table["sequence"].map(
                hierarchy.generate_sequence_id
            )
        if not descriptor_table["id"].is_unique:
            raise AssertionError("Descriptor table contains duplicate sequence IDs.")

        for row in descriptor_table.itertuples(index=False):
            summary = {}
            for descriptor in SELECTED_DESCRIPTOR_COLUMNS:
                value = _native(getattr(row, descriptor))
                if value is not None:
                    summary[descriptor] = round(float(value), DESCRIPTOR_DECIMALS)
            descriptor_summaries[str(row.id)] = summary
        descriptor_ids = set(descriptor_summaries)

    numerical_root = cfg.numerical_representation_root
    embedding_sets: dict[str, set[str]] = {}
    embedding_paths: dict[str, str] = {}
    embedding_root = numerical_root / "sylphy_embedding"
    if embedding_root.is_dir():
        for model_dir in sorted(path for path in embedding_root.iterdir() if path.is_dir()):
            full_data = model_dir / "full_data.csv"
            if not full_data.is_file():
                continue
            embedding_sets[model_dir.name] = _read_id_set(full_data, hierarchy)
            embedding_paths[model_dir.name] = str(full_data.relative_to(cfg.repo_root))

    one_hot_path = numerical_root / "sylphy_one_hot" / "one_hot" / "full_data.csv"
    one_hot_ids = _read_id_set(one_hot_path, hierarchy) if one_hot_path.is_file() else set()

    metadata = {
        "descriptors": {
            "available": descriptor_path.is_file(),
            "path": (
                str(descriptor_path.relative_to(cfg.repo_root))
                if descriptor_path.is_file()
                else str(descriptor_path)
            ),
            "feature_count": len(descriptor_features),
            "features": descriptor_features,
            "selected_for_sequence_cards": list(SELECTED_DESCRIPTOR_COLUMNS),
            "sequence_card_decimal_places": DESCRIPTOR_DECIMALS,
            "covered_sequences": len(descriptor_ids),
        },
        "one_hot": {
            "available": one_hot_path.is_file(),
            "path": (
                str(one_hot_path.relative_to(cfg.repo_root))
                if one_hot_path.is_file()
                else str(one_hot_path)
            ),
            "covered_sequences": len(one_hot_ids),
        },
        "embeddings": {
            model: {
                "path": embedding_paths[model],
                "covered_sequences": len(identifiers),
            }
            for model, identifiers in embedding_sets.items()
        },
    }
    return descriptor_summaries, one_hot_ids, embedding_sets, metadata


def _source_records(source_groups, key: tuple[str, str]) -> list[dict]:
    if key not in source_groups.indices:
        return []
    records = []
    for row in source_groups.get_group(key).itertuples(index=False):
        labels = getattr(row, "observed_labels")
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        records.append(
            {
                "source": str(row.source),
                "status": str(row.source_status),
                "observed_labels": [int(value) for value in labels],
            }
        )
    return sorted(records, key=lambda item: item["source"])


def _meaningful_targets(raw_categories: set[str]) -> list[str]:
    targets = set()
    for raw_category in raw_categories:
        normalized = str(raw_category).strip().lower()
        if normalized in {"", "no information", "no_information", "999"}:
            continue
        targets.update(
            part.strip()
            for part in normalized.split(",")
            if part.strip()
        )
    return sorted(targets)


def _source_counts(evidence_row: pd.Series) -> dict[str, int]:
    counts = {
        "positive": int(evidence_row.get("n_positive_sources", 0)),
        "negative": int(evidence_row.get("n_negative_sources", 0)),
        "unlabeled": int(evidence_row.get("n_unlabeled_sources", 0)),
        "conflicting": int(evidence_row.get("n_conflicting_sources", 0)),
    }
    return {name: value for name, value in counts.items() if value > 0}


def _schema(endpoints: list[str]) -> dict:
    statuses = ["positive", "negative", "ambiguous", "unlabeled", "no_information"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://maomao-resource.org/schemas/sequence-card-1.0.0.json",
        "title": "MAOMAO sequence card",
        "type": "object",
        "required": [
            "schema_version",
            "resource_version",
            "id",
            "sequence",
            "length",
            "activity_summary",
            "evidence_counts",
            "ontology",
            "negative_evidence",
            "toxicity_targets",
            "toxicity_properties",
            "physicochemical_summary",
        ],
        "properties": {
            "schema_version": {"const": CARD_SCHEMA_VERSION},
            "resource_version": {"type": "string"},
            "id": {"type": "string", "pattern": "^sha256_[0-9a-f]{64}$"},
            "sequence": {"type": "string", "pattern": "^[ACDEFGHIKLMNPQRSTVWY]+$"},
            "length": {"type": "integer", "minimum": 1},
            "activity_summary": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": False,
                "properties": {
                    status: {
                        "type": "array",
                        "items": {"enum": endpoints},
                        "uniqueItems": True,
                    }
                    for status in statuses
                },
            },
            "evidence_counts": {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/sourceCounts"},
            },
            "ontology": {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/ontologyEffect"},
            },
            "negative_evidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "contributes_to_ambiguity": {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/$defs/negativeEvidence"},
                    },
                    "supports_negative_status": {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/$defs/negativeEvidence"},
                    },
                },
            },
            "toxicity_targets": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
            },
            "toxicity_properties": {
                "type": "array",
                "items": {"$ref": "#/$defs/measurement"},
            },
            "physicochemical_summary": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    descriptor: {"type": "number"}
                    for descriptor in SELECTED_DESCRIPTOR_COLUMNS
                },
            },
        },
        "additionalProperties": False,
        "$defs": {
            "sourceCounts": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": False,
                "properties": {
                    "positive": {"type": "integer", "minimum": 1},
                    "negative": {"type": "integer", "minimum": 1},
                    "unlabeled": {"type": "integer", "minimum": 1},
                    "conflicting": {"type": "integer", "minimum": 1},
                },
            },
            "ontologyEffect": {
                "type": "object",
                "required": ["support_from", "direct_status", "final_status"],
                "additionalProperties": False,
                "properties": {
                    "support_from": {
                        "type": "array",
                        "items": {"enum": endpoints},
                        "uniqueItems": True,
                    },
                    "direct_status": {"enum": statuses},
                    "final_status": {"enum": statuses},
                },
            },
            "negativeEvidence": {
                "type": "object",
                "required": ["sources"],
                "additionalProperties": False,
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/classifiedNegativeSource"},
                        "uniqueItems": True,
                    },
                },
            },
            "classifiedNegativeSource": {
                "type": "object",
                "required": ["source"],
                "additionalProperties": False,
                "properties": {
                    "source": {"type": "string"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                },
            },
            "measurement": {
                "type": "object",
                "required": ["type", "relation", "value", "unit"],
                "properties": {
                    "type": {"enum": ["HC50", "LC50", "LD50", "MHC"]},
                    "relation": {"enum": ["=", "<", ">", "<=", ">="]},
                    "value": {"type": "number"},
                    "error": {"type": "number"},
                    "unit": {"type": "string"},
                    "reported_value": {"type": ["string", "number"]},
                    "source": {"type": "string"},
                    "source_dataset": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    }


def _render_example_html(card: dict) -> str:
    status_by_endpoint = {
        endpoint: status
        for status, endpoint_list in card["activity_summary"].items()
        for endpoint in endpoint_list
    }
    activity_rows = []
    for endpoint, status in status_by_endpoint.items():
        counts = card["evidence_counts"].get(endpoint, {})
        count_text = ", ".join(f"{name}: {value}" for name, value in counts.items()) or "—"
        ontology = card["ontology"].get(endpoint)
        ontology_text = (
            f"{ontology['direct_status']} → {ontology['final_status']} "
            f"(support: {', '.join(ontology['support_from'])})"
            if ontology
            else "—"
        )
        activity_rows.append(
            "<tr>"
            f"<td>{escape(endpoint)}</td>"
            f"<td><span class='state state-{escape(status)}'>{escape(status)}</span></td>"
            f"<td>{escape(count_text)}</td>"
            f"<td>{escape(ontology_text)}</td>"
            "</tr>"
        )

    negative_rows = []
    for interpretation, endpoint_records in card["negative_evidence"].items():
        for endpoint, evidence in endpoint_records.items():
            for source_record in evidence["sources"]:
                negative_rows.append(
                    "<tr>"
                    f"<td>{escape(endpoint)}</td>"
                    f"<td>{escape(interpretation)}</td>"
                    f"<td>{escape(source_record['source'])}</td>"
                    f"<td>{escape(', '.join(source_record.get('categories', [])) or 'not classified')}</td>"
                    "</tr>"
                )
    if not negative_rows:
        negative_rows.append("<tr><td colspan='4'>No negative evidence is shown for the final card states.</td></tr>")

    target_rows = [
        "<tr>"
        f"<td>{escape(endpoint)}</td>"
        f"<td>{escape(', '.join(targets))}</td>"
        "</tr>"
        for endpoint, targets in card["toxicity_targets"].items()
    ]
    if not target_rows:
        target_rows.append(
            "<tr><td colspan='2'>No affected organism or biological target is reported.</td></tr>"
        )

    measurement_rows = []
    for measurement in card["toxicity_properties"]:
        error = (
            f" ± {measurement['error']}"
            if "error" in measurement
            else ""
        )
        measurement_rows.append(
            "<tr>"
            f"<td>{escape(measurement['type'])}</td>"
            f"<td>{escape(measurement['relation'])} {measurement['value']}{escape(error)}</td>"
            f"<td>{escape(measurement['unit'])}</td>"
            f"<td>{escape(str(measurement.get('source', '—')))}</td>"
            "</tr>"
        )
    if not measurement_rows:
        measurement_rows.append("<tr><td colspan='4'>No numerical toxicity property is linked.</td></tr>")

    descriptor_rows = [
        "<tr>"
        f"<td>{escape(descriptor)}</td>"
        f"<td>{value}</td>"
        "</tr>"
        for descriptor, value in card["physicochemical_summary"].items()
    ]
    if not descriptor_rows:
        descriptor_rows.append("<tr><td colspan='2'>No physicochemical summary is available.</td></tr>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MAOMAO sequence card</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#5d6678; --line:#d9deea; --accent:#3846a5; }}
    body {{ margin:0; font:15px/1.5 system-ui,sans-serif; color:var(--ink); background:#f4f6fb; }}
    main {{ max-width:1100px; margin:32px auto; padding:0 20px 40px; }}
    header, section {{ background:white; border:1px solid var(--line); border-radius:12px; padding:20px; margin-bottom:16px; }}
    h1, h2 {{ margin-top:0; }}
    code {{ overflow-wrap:anywhere; }}
    .sequence {{ font-family:ui-monospace,monospace; overflow-wrap:anywhere; color:var(--accent); }}
    table {{ width:100%; border-collapse:collapse; }}
    th, td {{ text-align:left; padding:9px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
    th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
    .state {{ font-weight:650; }}
    .state-positive {{ color:#087443; }} .state-negative {{ color:#a03333; }}
    .state-ambiguous {{ color:#9a6000; }} .state-no_information {{ color:var(--muted); }}
    dl {{ display:grid; grid-template-columns:max-content 1fr; gap:8px 16px; }}
    dt {{ font-weight:650; }} dd {{ margin:0; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>MAOMAO sequence card</h1>
    <p><code>{escape(card['id'])}</code></p>
    <p class="sequence">{escape(card['sequence'])}</p>
    <p>Length: {card['length']} residues · Resource: {escape(card['resource_version'])}</p>
  </header>
  <section>
    <h2>Activity summary</h2>
    <table><thead><tr><th>Endpoint</th><th>Final status</th><th>Direct-source counts</th><th>Ontology</th></tr></thead>
    <tbody>{''.join(activity_rows)}</tbody></table>
  </section>
  <section>
    <h2>Negative evidence</h2>
    <table><thead><tr><th>Endpoint</th><th>Interpretation</th><th>Source</th><th>Source category</th></tr></thead>
    <tbody>{''.join(negative_rows)}</tbody></table>
  </section>
  <section>
    <h2>Affected organisms / toxicity targets</h2>
    <table><thead><tr><th>Endpoint</th><th>Reported target</th></tr></thead>
    <tbody>{''.join(target_rows)}</tbody></table>
  </section>
  <section>
    <h2>Toxicity properties</h2>
    <table><thead><tr><th>Type</th><th>Value</th><th>Unit</th><th>Context</th></tr></thead>
    <tbody>{''.join(measurement_rows)}</tbody></table>
  </section>
  <section>
    <h2>Physicochemical summary</h2>
    <table><thead><tr><th>Descriptor</th><th>Value</th></tr></thead>
    <tbody>{''.join(descriptor_rows)}</tbody></table>
  </section>
</main>
</body>
</html>
"""


def find_sequence_card(
    cards_path: Path,
    *,
    identifier: str | None = None,
    sequence: str | None = None,
) -> dict:
    """Return one card from a JSONL or JSONL.GZ collection."""

    path = Path(cards_path).expanduser().resolve()
    if path.is_dir():
        compressed = path / "sequence_cards.jsonl.gz"
        uncompressed = path / "sequence_cards.jsonl"
        path = compressed if compressed.is_file() else uncompressed
    _require_file(path, "sequence-card collection")

    if (identifier is None) == (sequence is None):
        raise ValueError("Provide exactly one of identifier or sequence.")

    if sequence is not None:
        normalized = "".join(str(sequence).split()).upper()
        if not normalized:
            raise ValueError("The query sequence is empty after normalization.")
        identifier = f"sha256_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                card = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if card.get("id") == identifier:
                return card

    raise KeyError(f"No sequence card was found for ID: {identifier}")


def export_sequence_card(
    cards_path: Path,
    output_dir: Path,
    *,
    identifier: str | None = None,
    sequence: str | None = None,
) -> dict[str, Path]:
    """Extract one card as readable JSON and HTML files."""

    card = find_sequence_card(
        cards_path,
        identifier=identifier,
        sequence=sequence,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = str(card["id"])
    json_path = destination / f"{stem}.json"
    html_path = destination / f"{stem}.html"
    _atomic_json(json_path, card)
    html_path.write_text(_render_example_html(card), encoding="utf-8")
    return {"json": json_path, "html": html_path}


def _readme(resource_version: str) -> str:
    readme = """# MAOMAO Sequence Card Layer

**Resource version:** `{resource_version}`  
**Card schema version:** `{card_schema_version}`

The MAOMAO Sequence Card Layer provides one compact, sequence-level profile for each
normalized peptide represented in the resource.

Each sequence card combines information from multiple MAOMAO components into a single
human- and machine-readable object, including:

- final toxicity endpoint states;
- direct source-evidence counts;
- ontology-supported annotations;
- source-specific negative evidence;
- affected organism or toxicity-target categories;
- quantitative toxicity measurements;
- selected physicochemical descriptors.

The complete collection is distributed as:

```text
sequence_profiles/sequence_cards.jsonl.gz
```

with one JSON object per peptide sequence.

Sequence cards are intended as a **compact access and interpretation layer**. They do not
replace the canonical MAOMAO tables. Complete source-level evidence, full-precision
descriptor matrices, numerical representations, and benchmark partitions remain in their
corresponding resource files.

---

# Directory structure

```text
sequence_profiles/
├── CHECKSUMS.sha256
├── examples/
│   ├── sequence_card_example.html
│   └── sequence_card_example.json
├── metadata.json
├── README.md
├── selected_cards/
│   ├── <sequence_id>.html
│   └── <sequence_id>.json
├── sequence_activity_evidence.csv.gz
├── sequence_card_schema.json
├── sequence_cards.jsonl.gz
└── sequence_source_evidence.csv.gz
```

Files:

- `sequence_cards.jsonl.gz`: complete streaming card collection.
- `sequence_activity_evidence.csv.gz`: one detailed row per sequence and endpoint.
- `sequence_source_evidence.csv.gz`: one detailed row per sequence, endpoint, and source.
- `sequence_card_schema.json`: JSON Schema for one card.
- `metadata.json`: layer provenance, paths, coverage, and checksums.
- `examples/sequence_card_example.json`: one human-readable example.
- `examples/sequence_card_example.html`: presentation view of that example.
- `CHECKSUMS.sha256`: integrity checks for distributed files.

The `selected_cards/` directory contains convenience JSON and HTML exports for individually
selected sequences. These files do not replace the complete JSONL collection.

---

# Sequence-card structure

Each card contains the following top-level fields:

| Field | Meaning |
|---|---|
| `schema_version` | Version of the sequence-card JSON schema. |
| `resource_version` | MAOMAO resource version from which the card was generated. |
| `id` | Stable SHA-256 sequence identifier used across MAOMAO. |
| `sequence` | Normalized canonical amino-acid sequence. |
| `length` | Sequence length in amino-acid residues. |
| `activity_summary` | Final MAOMAO toxicity states grouped by status. |
| `evidence_counts` | Direct source-evidence counts for each endpoint. |
| `ontology` | Positive hierarchical support affecting parent endpoints. |
| `negative_evidence` | Source-specific negative evidence relevant to final negative or ambiguous states. |
| `toxicity_targets` | Affected organism or target categories supported by direct positive evidence. |
| `toxicity_properties` | Source-reported quantitative toxicity measurements. |
| `physicochemical_summary` | Four selected sequence-derived physicochemical descriptors. |

A complete card therefore provides a compact view of **what is known about a peptide,
how that information was supported, and which additional quantitative properties are
available**.

---

# Sequence identifiers

Every card uses the same stable sequence identifier used throughout MAOMAO:

```text
sha256_<64 lowercase hexadecimal characters>
```

The identifier is generated from the normalized uppercase peptide sequence.

For example:

```text
sequence:
LFGFLIKLIPSLFGALSNIGRNRNQ

id:
sha256_911187d1b6c9fb53d175badca4e92992527f10be1cab3717e30d0cc6004214aa
```

The same identifier can be used to connect a card to endpoint tables, descriptor matrices,
numerical representations, and benchmark resources.

---

# Toxicity endpoints and ontology

The MAOMAO toxicity hierarchy represented by the cards is:

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

Positive evidence can propagate upward through this hierarchy. For example:

```text
Anti-mammalian cells positive
            ↓
       Cytotoxic positive
            ↓
         Toxic positive
```

Likewise:

```text
Hemolytic positive
        ↓
Cytotoxic positive
        ↓
    Toxic positive
```

Only **positive evidence** is propagated. Negative, ambiguous, unlabeled, and
no-information states are not propagated as negative evidence. Direct ambiguity at a
parent endpoint is retained and cannot be overwritten by positive hierarchical support.

---

# Activity summary

`activity_summary` reports the **final MAOMAO state** assigned to every toxicity endpoint.

Example:

```json
"activity_summary": {
  "ambiguous": [
    "toxic",
    "cytotoxic"
  ],
  "positive": [
    "hemolytic",
    "cytolysis",
    "ichthyotoxic"
  ],
  "negative": [
    "anti_mammalian_cells",
    "neurotoxic"
  ],
  "no_information": [
    "embryotoxic"
  ]
}
```

The possible final states are:

| State | Original pivot code | Interpretation |
|---|---:|---|
| `positive` | `1` | Direct positive evidence or positive evidence supported through the ontology. |
| `negative` | `0` | Direct endpoint-specific negative evidence. |
| `ambiguous` | `2` | Conflicting or otherwise unresolved evidence for the same sequence-endpoint pair. |
| `unlabeled` | `3` | The sequence was retained from an endpoint-associated source without an explicit positive or negative assignment. |
| `no_information` | `999` | No annotation is available for the sequence-endpoint pair. |

`no_information` is **not equivalent to negative**. Similarly, an `unlabeled` peptide
should not automatically be interpreted as either positive or negative.

---

# Direct source-evidence counts

`evidence_counts` summarizes the number of **direct contributing sources** associated with
each evidence type before ontology propagation.

Example:

```json
"evidence_counts": {
  "toxic": {
    "positive": 22,
    "negative": 1
  },
  "hemolytic": {
    "positive": 15,
    "unlabeled": 1
  }
}
```

Possible count categories are:

| Count | Meaning |
|---|---|
| `positive` | Number of direct sources supporting a positive state. |
| `negative` | Number of direct sources supporting a negative state. |
| `unlabeled` | Number of sources containing the sequence without an explicit binary assignment. |
| `conflicting` | Number of sources containing internally conflicting evidence for the sequence-endpoint pair. |

These counts represent **provenance support, not votes**.

A larger number of positive sources does not automatically override a negative source. For
example:

```text
positive sources: 22
negative sources: 1
final state: ambiguous
```

is possible because MAOMAO preserves conflicting evidence rather than resolving labels
through majority voting.

---

# Ontology support

The `ontology` object is included for endpoints receiving positive support from more
specific endpoints in the hierarchy.

Example:

```json
"ontology": {
  "toxic": {
    "support_from": [
      "cytotoxic",
      "ichthyotoxic"
    ],
    "direct_status": "ambiguous",
    "final_status": "ambiguous"
  }
}
```

| Field | Meaning |
|---|---|
| `support_from` | Child endpoints supplying positive hierarchical support. |
| `direct_status` | Endpoint state before applying hierarchical support. |
| `final_status` | Endpoint state after applying the hierarchy. |

For example:

```text
ambiguous → ambiguous
support: cytotoxic, ichthyotoxic
```

means that positive child endpoints support the parent endpoint, but the parent already
contains direct ambiguous evidence. Because MAOMAO preserves parent-level ambiguity, the
hierarchy does not overwrite that ambiguity.

Another possible situation is:

```text
no_information → positive
support: hemolytic
```

which indicates a hierarchy-derived positive annotation.

The `ontology` field should therefore be interpreted as **evidence-propagation
provenance**, not as another independent toxicity measurement.

---

# Negative evidence

`negative_evidence` preserves negative-source information that is relevant to the final
MAOMAO state.

It is divided into two interpretations:

```text
supports_negative_status
contributes_to_ambiguity
```

## `supports_negative_status`

Used when direct negative evidence supports an endpoint whose final state remains negative.

Example:

```json
"supports_negative_status": {
  "neurotoxic": {
    "sources": [
      {
        "source": "BiToxNet",
        "categories": [
          "weak or unconfirmed negatives"
        ]
      }
    ]
  }
}
```

## `contributes_to_ambiguity`

Used when negative evidence participates in a conflicting sequence-endpoint annotation.

Example:

```json
"contributes_to_ambiguity": {
  "toxic": {
    "sources": [
      {
        "source": "iAMPCN",
        "categories": [
          "weak or unconfirmed negatives"
        ]
      }
    ]
  }
}
```

Negative-evidence categories describe the **provenance or construction characteristics of
the negative dataset**. They are not probability scores and should not be interpreted as
numerical confidence values.

Negative records superseded by a final positive hierarchy assignment are omitted from the
compact card but remain available in the companion audit tables.

---

# Affected organisms and toxicity targets

`toxicity_targets` summarizes organism or target categories associated with **direct
positive evidence for endpoints whose final state is positive**.

Example:

```json
"toxicity_targets": {
  "hemolytic": [
    "animals",
    "human"
  ]
}
```

The values should be interpreted as reported toxicity-target categories.

They are **not**:

- the organism from which the peptide originated;
- a prediction of every organism that the peptide may affect;
- evidence attached to negative annotations;
- evidence attached only through ontology propagation.

A broad category such as `human` or `animals` remains broad and should not be interpreted
as a specific species unless the source provides that level of information.

An endpoint may be final-positive while being absent from `toxicity_targets`. This means
that no meaningful target category was linked to the relevant direct positive source
evidence; it does not invalidate the positive endpoint.

---

# Quantitative toxicity properties

`toxicity_properties` contains quantitative toxicity measurements linked to a MAOMAO
sequence.

The current card schema supports:

```text
HC50
LC50
LD50
MHC
```

These measurement types are retained as **distinct source-reported quantities** and are not
silently converted into a single common endpoint.

## Measurement types

| Type | Interpretation |
|---|---|
| `HC50` | Source-reported concentration associated with 50% hemolysis. |
| `LC50` | Source-reported LC50 measurement retained using the terminology of the source data. |
| `LD50` | Source-reported LD50 measurement retained using the terminology of the source data. |
| `MHC` | Source-reported minimum hemolytic concentration. |

Because these quantities can represent different experimental definitions, values from
different measurement types should not automatically be pooled into a single numerical
target.

## Structure of one measurement

Example:

```json
{
  "type": "LC50",
  "relation": "=",
  "value": 1.9,
  "unit": "µM",
  "reported_value": "1.9",
  "source": "Horse",
  "source_dataset": "Hemolytik2.0_2026"
}
```

| Field | Meaning |
|---|---|
| `type` | Quantitative measurement type: `HC50`, `LC50`, `LD50`, or `MHC`. |
| `relation` | Relationship between the reported measurement and the numerical value. |
| `value` | Parsed numerical value for computational reuse. |
| `error` | Reported numerical uncertainty when an error term is available. Optional. |
| `unit` | Standardized reported measurement unit. |
| `reported_value` | Original parsed value expression retained for traceability. |
| `source` | Experimental context retained from the source dataset, such as the erythrocyte source. |
| `source_dataset` | Dataset from which the measurement was obtained. |

## Measurement relations

The schema supports:

| Relation | Interpretation |
|---|---|
| `=` | Point estimate reported at the given value. |
| `>` | Reported value is greater than the numerical threshold. |
| `<` | Reported value is lower than the numerical threshold. |
| `>=` | Reported value is greater than or equal to the threshold. |
| `<=` | Reported value is lower than or equal to the threshold. |

For example:

```text
> 200 µM
```

must not be treated as an exact measurement of `200 µM`. It represents a bounded or
censored observation.

Likewise, a measurement such as:

```text
0.6 ± 0.1 µM
```

may be represented with:

```text
value = 0.6
error = 0.1
unit = µM
```

The `error` field preserves the numerical uncertainty supplied by the processed source
record. The card does not infer an uncertainty definition that is not explicitly
available.

## Units

Quantitative measurements retain their standardized reported unit. Units are **not
automatically converted when building the cards**.

Therefore:

```text
10 µM
```

and:

```text
10 µg/mL
```

must not be treated as numerically equivalent.

Comparisons should preferably be restricted to measurements with compatible measurement
type, unit, and experimental context.

## Repeated measurements

If a sequence has multiple reported quantitative measurements, the card retains the
individual measurements rather than averaging them.

This preserves experimental heterogeneity and allows users to select an aggregation
strategy appropriate for their own analysis.

If no quantitative toxicity measurement is available, the card contains:

```json
"toxicity_properties": []
```

The absence of a quantitative value does not imply absence of toxicity.

---

# Physicochemical summary

Each card contains a compact summary of four sequence-derived descriptors:

```text
net_charge_pH
boman_index
fcr
aa_entropy
```

These values are **computed sequence descriptors, not experimental toxicity measurements**.

The full MAOMAO descriptor matrix contains 41 descriptors. Only four representative
descriptors are repeated in each sequence card to provide a compact physicochemical view.

| Descriptor | Meaning | General interpretation |
|---|---|---|
| `net_charge_pH` | Estimated net peptide charge at the pH used by the descriptor workflow. | Positive values indicate net cationic character, negative values net anionic character, and values near zero indicate approximately balanced charge. |
| `boman_index` | Sequence-derived interaction-propensity descriptor. | Larger values indicate greater estimated interaction propensity within the same descriptor implementation. It is not an experimentally measured binding affinity. |
| `fcr` | Fraction of charged residues in the peptide sequence. | The value represents the proportion of residues classified as charged; for example, `0.12` corresponds to approximately 12% charged residues. |
| `aa_entropy` | Amino-acid composition entropy used as a sequence-complexity descriptor. | Larger values generally indicate greater compositional diversity, whereas smaller values indicate composition dominated by fewer residue types. |

For example:

```json
"physicochemical_summary": {
  "net_charge_pH": 2.997662,
  "boman_index": 0.076,
  "fcr": 0.12,
  "aa_entropy": 3.258689
}
```

These descriptors should primarily be used for **comparative characterization across
sequences**. They are not direct toxicity scores and do not have universal thresholds
separating toxic from non-toxic peptides.

Values stored in cards are rounded to six decimal places. The canonical descriptor table
retains the complete 41-descriptor matrix at full precision.

---

# Experimental properties versus calculated descriptors

The distinction between these two card sections is important:

| Card section | Origin | Example | Interpretation |
|---|---|---|---|
| `toxicity_properties` | Source-reported quantitative evidence | `LC50 = 1.9 µM` | Quantitative toxicity measurement retained from a source dataset. |
| `physicochemical_summary` | Calculated from peptide sequence | `fcr = 0.12` | Sequence-derived descriptor computed by the MAOMAO descriptor workflow. |

A physicochemical descriptor should therefore not be interpreted as an experimentally
measured toxicity value.

Likewise, the absence of an `HC50`, `LC50`, `LD50`, or `MHC` value does not prevent a
peptide from having calculated physicochemical descriptors.

---

# How to interpret the distributed example card

The distributed example contains:

```text
LFGFLIKLIPSLFGALSNIGRNRNQ
```

with the following final states:

```text
toxic                  ambiguous
cytotoxic              ambiguous
hemolytic              positive
cytolysis              positive
ichthyotoxic           positive
anti_mammalian_cells   negative
neurotoxic             negative
embryotoxic            no_information
```

For `toxic`, the example reports:

```text
positive sources: 22
negative sources: 1
ontology support: cytotoxic, ichthyotoxic
final status: ambiguous
```

This illustrates an important MAOMAO rule: **hierarchical support does not erase direct
ambiguity**.

Similarly:

```text
anti_mammalian_cells = negative
```

is an endpoint-specific negative state. Negative evidence is not propagated upward to make
`cytotoxic` or `toxic` negative.

The same example contains:

```text
LC50 = 1.9 µM
Context = Horse
```

which is a source-reported quantitative measurement.

Its physicochemical summary includes:

```text
net_charge_pH = 2.997662
fcr = 0.12
```

which are calculated sequence properties and are conceptually distinct from the
quantitative LC50 value.

---

# Companion evidence tables

The compact card is designed for interpretation and convenient access. For detailed
analysis, use the companion tables.

## `sequence_activity_evidence.csv.gz`

This file contains one row per sequence and toxicity endpoint.

Principal fields include:

| Field | Meaning |
|---|---|
| `id` | Stable MAOMAO sequence identifier. |
| `sequence` | Normalized peptide sequence. |
| `endpoint` | MAOMAO toxicity endpoint. |
| `status` | Final endpoint status. |
| `status_before_hierarchy` | Status before positive ontology propagation. |
| `status_origin` | Origin of the final assignment. |
| `n_positive_sources` | Number of direct positive sources. |
| `n_negative_sources` | Number of direct negative sources. |
| `n_unlabeled_sources` | Number of unlabeled sources. |
| `n_conflicting_sources` | Number of internally conflicting sources. |
| `hierarchy_source` | Endpoint or endpoints providing hierarchical support. |
| `is_hierarchy_inferred` | Whether the final status was inferred through hierarchy propagation. |
| `hierarchy_blocked_by_ambiguity` | Whether direct parent ambiguity prevented hierarchy-derived positivity. |
| `has_hierarchical_conflict` | Whether hierarchical support conflicts with direct endpoint evidence. |
| `negative_evidence_categories` | Negative-dataset provenance categories associated with the endpoint. |
| `toxicity_target_categories` | Direct positive toxicity-target categories. |

## `sequence_source_evidence.csv.gz`

This file contains one row per sequence, endpoint, and source and should be used when a
user needs to identify which individual source supplied an annotation.

Important fields include:

```text
id
sequence
endpoint
source
source_status
observed_labels
negative_evidence_categories
toxicity_target_categories
```

---

# Reading the card collection

The complete collection is stored as compressed JSON Lines so users do not need to load
all cards into memory simultaneously.

Read the first card:

```python
import gzip
import json

with gzip.open("sequence_cards.jsonl.gz", "rt", encoding="utf-8") as handle:
    for line in handle:
        card = json.loads(line)
        print(card["id"])
        break
```

Inspect its final activity states:

```python
print(card["activity_summary"])
```

Inspect quantitative toxicity properties:

```python
print(card["toxicity_properties"])
```

Inspect the compact physicochemical profile:

```python
print(card["physicochemical_summary"])
```

---

# Exporting one selected sequence

Export one selected sequence as readable JSON and HTML from the repository root:

```bash
python notebooks_and_scripts/sequence_cards/export_sequence_card.py \
  --sequence LFGFLIKLIPSLFGALSNIGRNRNQ
```

The selected files are written to `sequence_profiles/selected_cards/`.

The HTML file is intended for human inspection, whereas the JSON file preserves the
machine-readable card structure.

---

# Example files

The `examples/` directory contains one representative sequence card in two formats:

```text
examples/
├── sequence_card_example.json
└── sequence_card_example.html
```

Use the JSON example when inspecting the schema or developing software against the cards.

Use the HTML example when a human-readable summary is preferred.

---

# Data interpretation rules

When using sequence cards, the following distinctions should be preserved:

- `positive`, `negative`, `ambiguous`, `unlabeled`, and `no_information` are distinct states.
- `no_information` must never be interpreted as a negative annotation.
- Direct source counts are provenance summaries and are not majority votes.
- Positive ontology support does not overwrite direct ambiguity.
- Negative annotations are endpoint-specific and are not propagated through the ontology.
- `toxicity_targets` describes reported affected-organism or target categories, not peptide origin.
- A missing target category does not invalidate a positive endpoint.
- `HC50`, `LC50`, `LD50`, and `MHC` are distinct source-reported quantitative metrics.
- Values reported with `>` or `<` are bounded observations and should not be treated as exact values.
- Units are preserved and are not automatically converted.
- Repeated quantitative measurements are retained rather than averaged in the card layer.
- `toxicity_properties` contains source-reported quantitative measurements.
- `physicochemical_summary` contains calculated sequence descriptors.
- The four card descriptors are a compact subset of the complete 41-descriptor MAOMAO matrix.
- Card-level descriptor values are rounded to six decimal places; the canonical descriptor table retains full precision.

---

# Integrity verification

Distributed files can be checked using:

```bash
sha256sum -c CHECKSUMS.sha256
```

The checksums help detect accidental changes to the distributed files.

---

# Relationship to the complete MAOMAO resource

The Sequence Card Layer is a compact cross-resource access layer.

It intentionally does not duplicate:

- the complete 41-descriptor matrix;
- one-hot encoded sequences;
- protein language model embeddings;
- complete source evidence;
- benchmark train/validation/test partitions.

These resources remain in their canonical MAOMAO layers and can be connected through the
stable sequence identifier.

For complete information about installation, resource construction, endpoint definitions,
computational workflows, data availability, licensing, and citation, consult the main
MAOMAO `README.md`.

---

# Recommended use

For interactive exploration of the card collection, see:

```text
04_sequence_card_exploration.ipynb
```

The notebook demonstrates how to:

- stream the JSONL collection;
- retrieve individual cards;
- inspect activity and ontology information;
- examine quantitative measurements and physicochemical descriptors;
- filter sequences by endpoint and final status.
"""
    return (
        readme
        .replace("{resource_version}", str(resource_version))
        .replace("{card_schema_version}", CARD_SCHEMA_VERSION)
    )


def _write_evidence_tables(
    output_dir: Path,
    evidence_long: pd.DataFrame,
    hierarchy_audit: pd.DataFrame,
    source_resolved: pd.DataFrame,
    negative_categories: dict[tuple[str, str], set[str]],
    negative_source_categories: dict[tuple[str, str], set[str]],
    target_categories: dict[tuple[str, str], set[str]],
    source_target_categories: dict[tuple[str, str], set[str]],
) -> tuple[Path, Path]:
    audit_before = {
        (str(row.id), str(row.endpoint)): str(row.status_before_hierarchy)
        for row in hierarchy_audit.itertuples(index=False)
    }

    activity = evidence_long.copy()
    activity.insert(
        activity.columns.get_loc("status") + 1,
        "status_before_hierarchy",
        [
            audit_before.get((str(row.id), str(row.endpoint)), str(row.status))
            for row in activity.itertuples(index=False)
        ],
    )
    activity["negative_evidence_categories"] = [
        "|".join(sorted(negative_categories.get((str(row.id), str(row.endpoint)), set())))
        for row in activity.itertuples(index=False)
    ]
    activity["toxicity_target_categories"] = [
        "|".join(_meaningful_targets(target_categories.get((str(row.id), str(row.endpoint)), set())))
        for row in activity.itertuples(index=False)
    ]
    activity_columns = [
        "id",
        "sequence",
        "endpoint",
        "status",
        "status_before_hierarchy",
        "status_origin",
        "n_positive_sources",
        "n_negative_sources",
        "n_unlabeled_sources",
        "n_conflicting_sources",
        "hierarchy_source",
        "is_hierarchy_inferred",
        "hierarchy_blocked_by_ambiguity",
        "has_hierarchical_conflict",
        "negative_evidence_categories",
        "toxicity_target_categories",
    ]
    activity = activity.loc[:, activity_columns]

    source = source_resolved.loc[
        :, ["id", "sequence", "endpoint", "source", "source_status", "observed_labels"]
    ].copy()
    source["observed_labels"] = source["observed_labels"].map(
        lambda labels: "|".join(str(int(value)) for value in labels)
    )
    source["negative_evidence_categories"] = [
        "|".join(
            sorted(
                negative_source_categories.get(
                    (_source_key(row.source), str(row.endpoint)),
                    set(),
                )
            )
        )
        if str(row.source_status) == "negative"
        else ""
        for row in source.itertuples(index=False)
    ]
    source["toxicity_target_categories"] = [
        "|".join(
            _meaningful_targets(
                source_target_categories.get(
                    (_source_key(row.source), str(row.endpoint)),
                    set(),
                )
            )
        )
        for row in source.itertuples(index=False)
    ]

    activity_path = output_dir / "sequence_activity_evidence.csv.gz"
    source_path = output_dir / "sequence_source_evidence.csv.gz"
    temporary_activity = output_dir / ".sequence_activity_evidence.csv.gz.tmp"
    temporary_source = output_dir / ".sequence_source_evidence.csv.gz.tmp"

    activity.to_csv(temporary_activity, index=False, compression="gzip")
    source.to_csv(temporary_source, index=False, compression="gzip")
    os.replace(temporary_activity, activity_path)
    os.replace(temporary_source, source_path)
    return activity_path, source_path


def build_sequence_cards(config: CardConfig) -> dict:
    """Build the complete sequence-card layer and return its metadata."""

    cfg = config.resolved()
    hierarchy = _load_hierarchy_module(cfg.repo_root)
    master, endpoints = _load_master(cfg, hierarchy)
    master_ids = set(master["id"].astype(str))

    evidence_long, hierarchy_audit, source_resolved = _rebuild_evidence(
        cfg,
        master,
        endpoints,
        hierarchy,
    )
    negative_categories, negative_paths = _read_annotation_matrices(
        cfg,
        hierarchy,
        master_ids,
        "sequence_negative_evidece.csv",
    )
    negative_source_categories, negative_source_metadata = (
        _load_negative_source_categories(cfg, endpoints)
    )
    target_categories, target_paths = _read_annotation_matrices(
        cfg,
        hierarchy,
        master_ids,
        "sequence_by_organism.csv",
    )
    source_target_categories, source_target_metadata = (
        _load_source_toxicity_targets(cfg, endpoints)
    )
    properties, property_metadata = _load_measurements(cfg, master_ids)
    descriptor_summaries, one_hot_ids, embedding_sets, representation_metadata = (
        _load_representations(cfg, hierarchy, endpoints)
    )
    descriptor_ids = set(descriptor_summaries)

    output_dir = cfg.output_dir
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    evidence_index = evidence_long.set_index(["id", "endpoint"], verify_integrity=True)
    audit_before = {
        (str(row.id), str(row.endpoint)): str(row.status_before_hierarchy)
        for row in hierarchy_audit.itertuples(index=False)
    }
    source_groups = source_resolved.groupby(["id", "endpoint"], sort=False)

    activity_evidence_path, source_evidence_path = _write_evidence_tables(
        output_dir,
        evidence_long,
        hierarchy_audit,
        source_resolved,
        negative_categories,
        negative_source_categories,
        target_categories,
        source_target_categories,
    )

    cards_path = output_dir / "sequence_cards.jsonl.gz"
    temporary_cards = cards_path.with_name(f".{cards_path.name}.tmp")
    best_card: dict | None = None
    best_score = -1
    card_count = 0
    unique_ids: set[str] = set()
    ids_with_positive_targets: set[str] = set()

    with gzip.open(temporary_cards, "wt", encoding="utf-8", compresslevel=6) as handle:
        for master_row in master.itertuples(index=False):
            identifier = str(master_row.id)
            activity_summary: dict[str, list[str]] = defaultdict(list)
            evidence_counts = {}
            ontology = {}
            negative_evidence: dict[str, dict[str, dict]] = defaultdict(dict)
            toxicity_targets = {}

            for endpoint in endpoints:
                evidence_row = evidence_index.loc[(identifier, endpoint)]
                key = (identifier, endpoint)
                before_status = audit_before.get(key, str(evidence_row["status"]))
                final_status = str(evidence_row["status"])
                activity_summary[final_status].append(endpoint)

                counts = _source_counts(evidence_row)
                if counts:
                    evidence_counts[endpoint] = counts

                hierarchy_sources = [
                    item
                    for item in str(evidence_row.get("hierarchy_source", "")).split("|")
                    if item
                ]
                if hierarchy_sources:
                    ontology[endpoint] = {
                        "support_from": hierarchy_sources,
                        "direct_status": before_status,
                        "final_status": final_status,
                    }

                sources = _source_records(source_groups, key)
                negative_sources = sorted(
                    record["source"]
                    for record in sources
                    if record["status"] == "negative"
                )
                if negative_sources and final_status in {
                    "negative",
                    "ambiguous",
                }:
                    interpretation = (
                        "supports_negative_status"
                        if final_status == "negative"
                        else "contributes_to_ambiguity"
                    )
                    classified_sources = []
                    for source in negative_sources:
                        source_record = {"source": source}
                        categories = sorted(
                            negative_source_categories.get(
                                (_source_key(source), endpoint),
                                set(),
                            )
                        )
                        if categories:
                            source_record["categories"] = categories
                        classified_sources.append(source_record)

                    negative_evidence[interpretation][endpoint] = {
                        "sources": classified_sources
                    }

                positive_sources = [
                    record["source"]
                    for record in sources
                    if record["status"] == "positive"
                ]
                positive_target_categories = set()
                for source in positive_sources:
                    positive_target_categories.update(
                        source_target_categories.get(
                            (_source_key(source), endpoint),
                            set(),
                        )
                    )
                targets = _meaningful_targets(positive_target_categories)
                if final_status == "positive" and targets:
                    toxicity_targets[endpoint] = targets

            card = {
                "schema_version": CARD_SCHEMA_VERSION,
                "resource_version": str(cfg.resource_version),
                "id": identifier,
                "sequence": str(master_row.sequence),
                "length": len(str(master_row.sequence)),
                "activity_summary": dict(activity_summary),
                "evidence_counts": evidence_counts,
                "ontology": ontology,
                "negative_evidence": {
                    interpretation: endpoint_records
                    for interpretation, endpoint_records in negative_evidence.items()
                },
                "toxicity_targets": toxicity_targets,
                "toxicity_properties": properties.get(identifier, []),
                "physicochemical_summary": descriptor_summaries.get(identifier, {}),
            }
            handle.write(json.dumps(card, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

            if identifier in unique_ids:
                raise AssertionError(f"Duplicate sequence card ID: {identifier}")
            unique_ids.add(identifier)
            card_count += 1
            if toxicity_targets:
                ids_with_positive_targets.add(identifier)

            score = (
                20 * len(card["toxicity_properties"])
                + 3 * sum(sum(counts.values()) for counts in evidence_counts.values())
                + 4 * sum(
                    len(endpoint_records)
                    for endpoint_records in negative_evidence.values()
                )
                + 2 * len(ontology)
                + sum(len(targets) for targets in toxicity_targets.values())
            )
            if score > best_score:
                best_card = card
                best_score = score

    os.replace(temporary_cards, cards_path)

    if card_count != len(master) or unique_ids != master_ids:
        raise AssertionError("The generated card set does not match the master pivot.")
    if best_card is None:
        raise AssertionError("No sequence cards were generated.")

    schema_path = output_dir / "sequence_card_schema.json"
    example_json_path = examples_dir / "sequence_card_example.json"
    example_html_path = examples_dir / "sequence_card_example.html"
    readme_path = output_dir / "README.md"
    metadata_path = output_dir / "metadata.json"
    checksum_path = output_dir / "CHECKSUMS.sha256"

    _atomic_json(schema_path, _schema(endpoints))
    _atomic_json(example_json_path, best_card)
    example_html_path.write_text(_render_example_html(best_card), encoding="utf-8")
    readme_path.write_text(_readme(str(cfg.resource_version)), encoding="utf-8")

    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "resource_version": str(cfg.resource_version),
        "sequence_card_schema_version": CARD_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "identifier_format": "sha256_<64_lowercase_hex>",
        "card_count": card_count,
        "endpoints": endpoints,
        "ontology": hierarchy.PARENT_CHILDREN,
        "activity_codebook": {str(code): status for status, code in STATUS_TO_CODE.items()},
        "semantics": {
            "activity_summary": "Endpoints grouped by final mutually exclusive status after hierarchy.",
            "ontology": (
                "Only endpoints receiving upward positive support; direct_status and final_status "
                "show whether the hierarchy changed the assignment."
            ),
            "negative_evidence": (
                "Shown in cards only when it supports a final negative state or contributes to ambiguity; "
                "each category is attached to its corresponding source. Negative records omitted from "
                "final-positive cards remain in the companion audits."
            ),
            "toxicity_targets": (
                "Affected organism or biological-target categories linked to direct positive source evidence "
                "for endpoints whose final status is positive; not peptide-origin organisms."
            ),
            "descriptors": (
                "Cards include four selected values rounded to six decimals; the canonical descriptor matrix "
                "retains all descriptors at full precision."
            ),
            "detailed_evidence": (
                "Source-level and sequence-endpoint audit fields are distributed as companion CSV files."
            ),
        },
        "inputs": {
            "master_pivot": str(cfg.master_path.relative_to(cfg.repo_root)),
            "integrated_endpoints": "processed_data/integrating_and_cleaning_data/",
            "negative_evidence_matrices": negative_paths,
            "negative_evidence_source_metadata": negative_source_metadata,
            "toxicity_target_matrices": target_paths,
            "toxicity_target_source_metadata": source_target_metadata,
            "toxicity_properties": property_metadata,
            "representations": representation_metadata,
        },
        "coverage": {
            "sequences_with_toxicity_properties": len(properties),
            "sequences_with_toxicity_targets": len(
                ids_with_positive_targets
            ),
            "sequences_with_descriptors": len(master_ids.intersection(descriptor_ids)),
            "sequences_with_one_hot": len(master_ids.intersection(one_hot_ids)),
            "embedding_models": {
                model: len(master_ids.intersection(identifiers))
                for model, identifiers in embedding_sets.items()
            },
        },
        "validation": {
            "master_ids_match_sequence_sha256": True,
            "evidence_matches_master_pivot": True,
            "one_card_per_master_sequence": True,
            "unique_card_ids": True,
        },
        "files": {},
    }

    files_before_metadata = [
        cards_path,
        activity_evidence_path,
        source_evidence_path,
        schema_path,
        readme_path,
        example_json_path,
        example_html_path,
    ]
    for path in files_before_metadata:
        metadata["files"][str(path.relative_to(output_dir))] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    _atomic_json(metadata_path, metadata)

    checksum_files = [
        cards_path,
        activity_evidence_path,
        source_evidence_path,
        schema_path,
        metadata_path,
        readme_path,
        example_json_path,
        example_html_path,
    ]
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(output_dir)}\n"
            for path in checksum_files
        ),
        encoding="utf-8",
    )

    with gzip.open(cards_path, "rt", encoding="utf-8") as handle:
        first_card = json.loads(next(handle))
    required_keys = {
        "schema_version",
        "resource_version",
        "id",
        "sequence",
        "length",
        "activity_summary",
        "evidence_counts",
        "ontology",
        "negative_evidence",
        "toxicity_targets",
        "toxicity_properties",
        "physicochemical_summary",
    }
    summarized_endpoints = {
        endpoint
        for endpoint_list in first_card["activity_summary"].values()
        for endpoint in endpoint_list
    }
    if set(first_card) != required_keys or summarized_endpoints != set(endpoints):
        raise AssertionError("Generated card structure does not match the declared schema.")

    return {
        "output_dir": output_dir,
        "metadata": metadata,
        "example_card": best_card,
        "files": [*checksum_files, checksum_path],
    }
