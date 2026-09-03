from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import json
import re
import hashlib
import numpy as np
import pandas as pd

SEQUENCE_COL = "sequence"
MISSING_VALUE_OLD = 999
VALID_SOURCE_LABELS = {0, 1, 2, MISSING_VALUE_OLD}

ENDPOINTS = [
    "toxic",
    "cytotoxic",
    "hemolytic",
    "cytolysis",
    "neurotoxic",
    "embryotoxic",
    "ichthyotoxic",
    "anti_mammalian_cells"
]

ENDPOINT_FOLDER_MAP = {
    "toxic": ["toxic"],
    "cytotoxic": ["cytotoxic"],
    "hemolytic": ["hemolytic"],
    "cytolysis": ["cytolysis", "cytolytic"],
    "neurotoxic": ["neurotoxic"],
    "embryotoxic": ["embryotoxic"],
    "ichthyotoxic": ["ichthyotoxic"],
    "anti_mammalian_cells": ["anti_mammalian_cells"],
}

PARENT_CHILDREN = {
    "cytotoxic": ["hemolytic", "cytolysis"],
    "toxic": ["cytotoxic", "neurotoxic", "embryotoxic", "ichthyotoxic"],
    "anti_mammalian_cells": ["anti_mammalian_cells"],
}

STATUS_FILES = {
    "positive": "positive.csv",
    "negative": "negative.csv",
    "ambiguous": "ambiguous_data.csv",
    "unlabeled": "only_unlabel.csv",
}

AUXILIARY_COLUMNS = {
    SEQUENCE_COL,
    "task",
    "label",
    "status",
    "Category_pbb",
    "category_pbb",
    "counts_0",
    "counts_1",
    "counts_2",
    "counts_unlabel",
    "counts_unknown",
    "count_0",
    "count_1",
    "count_2",
    "count_unlabel",
    "count_unknown",
    "positive",
    "negative",
    "exclusive_0",
    "exclusive_1",
    "only_unlabel",
    "percentage_0",
    "percentage_1",
    "positive_support",
    "is_canon",
    "filter_length",
    "length",
}

@dataclass(frozen=True)
class Config:
    input_root: Path
    output_root: Path
    min_length: int = 5
    max_length: int = 70
    canonical_residues: str = "ACDEFGHIKLMNPQRSTVWY"
    include_full_cross_product: bool = True


def normalize_sequence(value):
    if pd.isna(value):
        return pd.NA
    sequence = re.sub(r"\s+", "", str(value)).upper()
    return sequence or pd.NA


def generate_sequence_id(sequence: str) -> str:
    """
    Generate a stable sequence-derived identifier using SHA-256.

    The identifier is calculated from the UTF-8 representation of the
    normalized peptide sequence.
    """
    digest = hashlib.sha256(
        sequence.encode("utf-8")
    ).hexdigest()

    return f"sha256_{digest}"


def sequence_qc(sequence, cfg: Config):
    if pd.isna(sequence):
        return False, "missing"
    if not (cfg.min_length <= len(sequence) <= cfg.max_length):
        return False, "length"
    if not set(sequence).issubset(set(cfg.canonical_residues)):
        return False, "noncanonical"
    return True, "pass"


def parse_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }).fillna(False).astype(bool)


def numeric_or_nan(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def identify_source_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        if col in AUXILIARY_COLUMNS or col.startswith("Unnamed:"):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        non_missing = values.dropna()
        if non_missing.empty:
            continue
        if not np.isclose(non_missing, np.round(non_missing)).all():
            continue
        observed = set(np.round(non_missing).astype(int).unique().tolist())
        if observed and observed.issubset(VALID_SOURCE_LABELS):
            candidates.append(col)
    return candidates


def discover_input_files(cfg: Config) -> pd.DataFrame:
    rows = []
    for endpoint, folders in ENDPOINT_FOLDER_MAP.items():
        for folder_name in folders:
            folder = cfg.input_root / folder_name
            for declared_status, filename in STATUS_FILES.items():
                path = folder / filename
                rows.append({
                    "endpoint": endpoint,
                    "input_endpoint_folder": folder_name,
                    "declared_status": declared_status,
                    "filename": filename,
                    "path": str(path),
                    "exists": path.exists(),
                })
    return pd.DataFrame(rows)


def infer_row_status(df: pd.DataFrame, declared_status: str) -> pd.Series:
    """Use the stage-2 flags as the source of truth, not the filename."""
    index = df.index
    p = parse_boolean(df["positive"]) if "positive" in df else pd.Series(False, index=index)
    n = parse_boolean(df["negative"]) if "negative" in df else pd.Series(False, index=index)
    u = parse_boolean(df["only_unlabel"]) if "only_unlabel" in df else pd.Series(False, index=index)

    counts_1 = numeric_or_nan(df["counts_1"]) if "counts_1" in df else pd.Series(np.nan, index=index)
    counts_0 = numeric_or_nan(df["counts_0"]) if "counts_0" in df else pd.Series(np.nan, index=index)

    # Contradictory row flags are ambiguous.
    contradictory = p & n
    inferred = pd.Series(pd.NA, index=index, dtype="string")
    inferred.loc[contradictory] = "ambiguous"
    inferred.loc[p & ~n] = "positive"
    inferred.loc[n & ~p] = "negative"
    inferred.loc[u & ~p & ~n] = "unlabeled"

    unresolved = inferred.isna()
    has_mixed_counts = counts_1.gt(0) & counts_0.gt(0)
    inferred.loc[unresolved & has_mixed_counts] = "ambiguous"

    # When internal stage-2 flags are unavailable, use file membership.
    inferred = inferred.fillna(declared_status)
    return inferred.astype("string")


def extract_positive_fraction(df: pd.DataFrame) -> pd.Series:
    if "percentage_1" in df.columns:
        value = numeric_or_nan(df["percentage_1"]) / 100.0
        return value.clip(0, 1)
    if {"counts_1", "counts_0"}.issubset(df.columns):
        p = numeric_or_nan(df["counts_1"])
        n = numeric_or_nan(df["counts_0"])
        den = p + n
        return p.div(den.where(den > 0)).clip(0, 1)
    return pd.Series(np.nan, index=df.index, dtype="float64")


def read_status_file(path: Path, endpoint: str, folder_name: str, declared_status: str, cfg: Config):
    df = pd.read_csv(path, low_memory=False)
    if SEQUENCE_COL not in df.columns:
        raise ValueError(f"Missing required column '{SEQUENCE_COL}' in {path}")

    df = df.copy()
    df[SEQUENCE_COL] = df[SEQUENCE_COL].map(normalize_sequence)
    df = df[df[SEQUENCE_COL].notna()].copy()

    qc = df[SEQUENCE_COL].map(lambda s: sequence_qc(s, cfg)[0])
    df = df[qc].copy()

    # Detect only original source columns, before derived audit columns are added.
    source_cols = identify_source_columns(df)

    df["endpoint"] = endpoint
    df["input_endpoint_folder"] = folder_name
    df["declared_status"] = declared_status
    df["inferred_row_status"] = infer_row_status(df, declared_status)
    df["status_file_mismatch"] = df["inferred_row_status"].ne(declared_status)
    df["input_file"] = path.name
    df["positive_evidence_fraction_reported"] = extract_positive_fraction(df)
    df["support_category"] = df.get(
        "Category_pbb",
        pd.Series(pd.NA, index=df.index, dtype="object"),
    )

    row_cols = [
        SEQUENCE_COL,
        "endpoint",
        "input_endpoint_folder",
        "declared_status",
        "inferred_row_status",
        "status_file_mismatch",
        "input_file",
        "positive_evidence_fraction_reported",
        "support_category",
    ]
    for col in [
        "counts_1",
        "counts_0",
        "counts_2",
        "counts_unlabel",
        "counts_unknown",
        "positive",
        "negative",
        "only_unlabel",
    ]:
        if col in df.columns:
            row_cols.append(col)
    row_records = df[row_cols].copy()

    if source_cols:
        source = df[
            [
                SEQUENCE_COL,
                "endpoint",
                "input_endpoint_folder",
                "inferred_row_status",
                "input_file",
            ]
            + source_cols
        ].melt(
            id_vars=[
                SEQUENCE_COL,
                "endpoint",
                "input_endpoint_folder",
                "inferred_row_status",
                "input_file",
            ],
            value_vars=source_cols,
            var_name="source",
            value_name="source_label",
        )
        source["source_label"] = pd.to_numeric(
            source["source_label"], errors="coerce"
        ).astype("Int64")
        source = source[source["source_label"].isin([0, 1, 2])].copy()
        source["evidence_origin"] = "source_column"
    else:
        fallback_map = {
            "positive": 1,
            "negative": 0,
            "unlabeled": 2,
            "ambiguous": pd.NA,
        }
        source = df[
            [
                SEQUENCE_COL,
                "endpoint",
                "input_endpoint_folder",
                "inferred_row_status",
                "input_file",
            ]
        ].copy()
        source["source"] = f"file:{folder_name}/{path.name}"
        source["source_label"] = source["inferred_row_status"].map(fallback_map).astype("Int64")
        source = source[source["source_label"].notna()].copy()
        source["evidence_origin"] = "file_membership_fallback"

    return source, row_records, source_cols


def collect_inputs(cfg: Config):
    manifest = discover_input_files(cfg)
    source_frames = []
    row_frames = []
    source_column_report = []

    for record in manifest[manifest["exists"]].to_dict("records"):
        source, rows, source_cols = read_status_file(
            Path(record["path"]),
            record["endpoint"],
            record["input_endpoint_folder"],
            record["declared_status"],
            cfg,
        )
        source_frames.append(source)
        row_frames.append(rows)
        source_column_report.append({
            **record,
            "n_source_columns": len(source_cols),
            "source_columns": "|".join(source_cols),
        })

    if not row_frames:
        raise FileNotFoundError(
            f"No compatible endpoint files were found under {cfg.input_root}"
        )

    source_raw = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
    row_raw = pd.concat(row_frames, ignore_index=True)
    return source_raw, row_raw, manifest, pd.DataFrame(source_column_report)


def resolve_source_evidence(source_raw: pd.DataFrame):
    if source_raw.empty:
        return source_raw.copy(), pd.DataFrame()

    keys = [SEQUENCE_COL, "endpoint", "source"]
    grouped = (
        source_raw.groupby(keys, dropna=False)["source_label"]
        .agg(lambda values: sorted(set(int(x) for x in values.dropna())))
        .reset_index(name="observed_labels")
    )

    def resolve(labels):
        labels = set(labels)
        if labels == {1}:
            return "positive"
        if labels == {0}:
            return "negative"
        if labels == {2}:
            return "unlabeled"
        if not labels:
            return "no_information"
        return "conflicting"

    grouped["source_status"] = grouped["observed_labels"].map(resolve)
    conflicts = grouped[grouped["source_status"].eq("conflicting")].copy()
    return grouped, conflicts


def aggregate_row_membership(row_raw: pd.DataFrame):
    """Resolve possible stale/overlapping status files without trusting filenames."""
    status_sets = (
        row_raw.groupby([SEQUENCE_COL, "endpoint"])["inferred_row_status"]
        .agg(lambda values: sorted(set(values.dropna().astype(str))))
        .reset_index(name="observed_row_statuses")
    )

    status_sets["has_direct_positive"] = status_sets["observed_row_statuses"].map(lambda x: "positive" in x)
    status_sets["has_direct_negative"] = status_sets["observed_row_statuses"].map(lambda x: "negative" in x)
    status_sets["has_direct_ambiguous"] = status_sets["observed_row_statuses"].map(lambda x: "ambiguous" in x)
    status_sets["has_direct_unlabeled"] = status_sets["observed_row_statuses"].map(lambda x: "unlabeled" in x)

    status_sets["has_status_membership_conflict"] = (
        status_sets[[
            "has_direct_positive",
            "has_direct_negative",
            "has_direct_ambiguous",
            "has_direct_unlabeled",
        ]].sum(axis=1) > 1
    )

    # A direct ambiguity or a conflict between input subsets has priority over
    # positive, negative, and unlabeled assignments for the same endpoint.
    status_sets["status_before_hierarchy"] = np.select(
        [
            (
                status_sets["has_direct_ambiguous"]
                | status_sets["has_status_membership_conflict"]
            ),
            status_sets["has_direct_positive"],
            status_sets["has_direct_negative"],
            status_sets["has_direct_unlabeled"],
        ],
        ["ambiguous", "positive", "negative", "unlabeled"],
        default="no_information",
    )
    return status_sets


def aggregate_source_counts(source_resolved: pd.DataFrame):
    if source_resolved.empty:
        return pd.DataFrame(columns=[SEQUENCE_COL, "endpoint"])

    counts = (
        source_resolved.assign(value=1)
        .pivot_table(
            index=[SEQUENCE_COL, "endpoint"],
            columns="source_status",
            values="value",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for col in ["positive", "negative", "unlabeled", "conflicting"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts.rename(columns={
        "positive": "n_positive_sources",
        "negative": "n_negative_sources",
        "unlabeled": "n_unlabeled_sources",
        "conflicting": "n_conflicting_sources",
    })
    return counts


def aggregate_reported_support(row_raw: pd.DataFrame):
    def first_valid(values: Iterable):
        for value in values:
            if pd.notna(value):
                return value
        return pd.NA

    return (
        row_raw.groupby([SEQUENCE_COL, "endpoint"], dropna=False)
        .agg(
            positive_evidence_fraction_reported=(
                "positive_evidence_fraction_reported",
                "max",
            ),
            support_category=("support_category", first_valid),
            input_files=("input_file", lambda x: "|".join(sorted(set(map(str, x))))),
            input_endpoint_folders=(
                "input_endpoint_folder",
                lambda x: "|".join(sorted(set(map(str, x)))),
            ),
            status_file_mismatch=("status_file_mismatch", "max"),
        )
        .reset_index()
    )


def build_direct_evidence(row_raw: pd.DataFrame, source_resolved: pd.DataFrame):
    membership = aggregate_row_membership(row_raw)
    source_counts = aggregate_source_counts(source_resolved)
    support = aggregate_reported_support(row_raw)

    direct = membership.merge(source_counts, on=[SEQUENCE_COL, "endpoint"], how="left")
    direct = direct.merge(support, on=[SEQUENCE_COL, "endpoint"], how="left")

    count_cols = [
        "n_positive_sources",
        "n_negative_sources",
        "n_unlabeled_sources",
        "n_conflicting_sources",
    ]
    for col in count_cols:
        direct[col] = direct[col].fillna(0).astype("int64")

    den = direct["n_positive_sources"] + direct["n_negative_sources"]
    direct["positive_evidence_fraction"] = direct["n_positive_sources"].div(
        den.where(den > 0)
    )
    direct["positive_evidence_fraction"] = direct[
        "positive_evidence_fraction"
    ].fillna(direct["positive_evidence_fraction_reported"])

    direct["has_direct_evidence"] = True
    direct["is_ambiguous"] = (
        direct["has_direct_ambiguous"]
        | direct["has_status_membership_conflict"]
        | (
            direct["n_positive_sources"].gt(0)
            & direct["n_negative_sources"].gt(0)
        )
        | direct["n_conflicting_sources"].gt(0)
    )

    # Source-level conflicts can reveal ambiguity even when the sequence did
    # not arrive through ambiguous_data.csv. The final direct state must still
    # be ambiguous for that same endpoint.
    direct.loc[
        direct["is_ambiguous"],
        "status_before_hierarchy",
    ] = "ambiguous"

    return direct


def complete_grid(direct: pd.DataFrame, cfg: Config):
    sequences = pd.Index(sorted(direct[SEQUENCE_COL].unique()), name=SEQUENCE_COL)
    if cfg.include_full_cross_product:
        grid = pd.MultiIndex.from_product(
            [sequences, ENDPOINTS], names=[SEQUENCE_COL, "endpoint"]
        ).to_frame(index=False)
        out = grid.merge(direct, on=[SEQUENCE_COL, "endpoint"], how="left")
    else:
        out = direct.copy()

    out["status_before_hierarchy"] = out["status_before_hierarchy"].fillna("no_information")
    out["has_direct_evidence"] = out["has_direct_evidence"].astype("boolean").fillna(False).astype(bool)
    out["is_ambiguous"] = out["is_ambiguous"].astype("boolean").fillna(False).astype(bool)
    for col in [
        "has_direct_positive",
        "has_direct_negative",
        "has_direct_ambiguous",
        "has_direct_unlabeled",
        "has_status_membership_conflict",
        "status_file_mismatch",
    ]:
        out[col] = out[col].astype("boolean").fillna(False).astype(bool)
    for col in [
        "n_positive_sources",
        "n_negative_sources",
        "n_unlabeled_sources",
        "n_conflicting_sources",
    ]:
        out[col] = out[col].fillna(0).astype("int64")
    return out


def apply_positive_only_hierarchy(
    evidence: pd.DataFrame,
):
    """
    Apply positive-only hierarchical propagation.

    A positive child provides positive support to its parent. However, a
    parent with direct endpoint-specific ambiguity remains ambiguous.

    Positive support is propagated transitively even when an intermediate
    endpoint remains ambiguous. For example:

        Hemolytic positive
        -> positive support for Cytotoxic
        -> positive support for Toxic

    Cytotoxic may remain ambiguous, while Toxic can still become positive when
    Toxic itself is not ambiguous.
    """
    before = evidence.pivot(
        index=SEQUENCE_COL,
        columns="endpoint",
        values="status_before_hierarchy",
    ).reindex(columns=ENDPOINTS)

    status = before.copy()

    # Tracks biological positive support independently from the final,
    # mutually exclusive state assigned to each endpoint.
    positive_support = before.eq("positive").copy()

    hierarchy_source = pd.DataFrame(
        "",
        index=before.index,
        columns=ENDPOINTS,
    )
    hierarchy_source.columns.name = "endpoint"

    # Parent order is important because positive support is propagated
    # transitively from Cytotoxic to Toxic.
    for parent in ["cytotoxic", "toxic"]:
        children = PARENT_CHILDREN[parent]

        child_positive_support = positive_support[children]
        any_positive_support = child_positive_support.any(axis=1)

        origins = child_positive_support.apply(
            lambda row: "|".join(
                row.index[row].tolist()
            ),
            axis=1,
        )

        hierarchy_source.loc[
            any_positive_support,
            parent,
        ] = origins[any_positive_support]

        # Keep the positive-support closure even when the final state of the
        # parent remains ambiguous.
        positive_support.loc[
            any_positive_support,
            parent,
        ] = True

        parent_is_ambiguous = before[parent].eq("ambiguous")
        hierarchy_can_update = (
            any_positive_support
            & ~parent_is_ambiguous
        )

        status.loc[
            hierarchy_can_update,
            parent,
        ] = "positive"

    status_long = (
        status
        .stack(future_stack=True)
        .rename("status")
        .reset_index()
    )
    source_long = (
        hierarchy_source
        .stack(future_stack=True)
        .rename("hierarchy_source")
        .reset_index()
    )

    result = evidence.merge(
        status_long,
        on=[SEQUENCE_COL, "endpoint"],
        how="left",
    )
    result = result.merge(
        source_long,
        on=[SEQUENCE_COL, "endpoint"],
        how="left",
    )

    result["has_hierarchical_positive_evidence"] = (
        result["hierarchy_source"]
        .fillna("")
        .ne("")
    )

    result["hierarchy_blocked_by_ambiguity"] = (
        result["has_hierarchical_positive_evidence"]
        & result["status_before_hierarchy"].eq("ambiguous")
    )

    result["is_hierarchy_inferred"] = (
        result["has_hierarchical_positive_evidence"]
        & result["status_before_hierarchy"].ne("positive")
        & result["status_before_hierarchy"].ne("ambiguous")
        & result["status"].eq("positive")
    )

    result["has_hierarchical_conflict"] = (
        result["has_hierarchical_positive_evidence"]
        & result["status_before_hierarchy"].isin(
            ["negative", "ambiguous"]
        )
    )

    # Endpoint-specific ambiguity and the final canonical state are now fully
    # synchronized. Hierarchy never removes an ambiguous assignment.
    result["is_ambiguous"] = result[
        "status"
    ].eq("ambiguous")

    result["label"] = (
        result["status"]
        .map(
            {
                "positive": 1,
                "negative": 0,
            }
        )
        .astype("Int64")
    )

    result["status_origin"] = np.select(
        [
            result["hierarchy_blocked_by_ambiguity"],
            (
                result["is_hierarchy_inferred"]
                & result["has_direct_evidence"]
            ),
            result["is_hierarchy_inferred"],
            (
                result["status"].eq("positive")
                & result[
                    "has_hierarchical_positive_evidence"
                ]
            ),
            result["has_direct_evidence"],
        ],
        [
            "direct_ambiguous_hierarchy_blocked",
            "direct_and_hierarchy",
            "hierarchy",
            "direct_and_hierarchy",
            "direct",
        ],
        default="no_information",
    )

    hierarchy_audit = result.loc[
        result["has_hierarchical_positive_evidence"],
        [
            SEQUENCE_COL,
            "endpoint",
            "status_before_hierarchy",
            "status",
            "hierarchy_source",
            "is_hierarchy_inferred",
            "hierarchy_blocked_by_ambiguity",
            "has_hierarchical_conflict",
            "is_ambiguous",
        ],
    ].copy()

    hierarchy_audit.insert(
        0,
        "id",
        hierarchy_audit[SEQUENCE_COL].map(
            generate_sequence_id
        ),
    )

    canonical = result.drop(
        columns=["status_before_hierarchy"]
    )

    return canonical, hierarchy_audit


def encode_endpoint_state(
    status: pd.Series,
) -> pd.Series:
    """
    Encode the final mutually exclusive endpoint state.

    0   = negative
    1   = positive
    2   = ambiguous
    3   = unlabeled
    999 = no information
    """
    encoding = {
        "negative": 0,
        "positive": 1,
        "ambiguous": 2,
        "unlabeled": 3,
        "no_information": 999,
    }

    encoded = status.astype("string").map(
        encoding
    )

    if encoded.isna().any():
        unknown = sorted(
            status.loc[encoded.isna()]
            .dropna()
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Unknown endpoint states: {unknown}"
        )

    return encoded.astype("int64")


def build_wide_pivot(evidence_long: pd.DataFrame):
    """
    Build the main user-facing sequence file.

    Output columns:
    - sequence
    - one coded column per endpoint

    No sequence-length, ambiguity-boolean, support, source-count, or hierarchy
    columns are included in this file.
    """
    sequences = pd.Index(
        sorted(evidence_long[SEQUENCE_COL].unique()),
        name=SEQUENCE_COL,
    )
    wide = pd.DataFrame(index=sequences)

    for endpoint in ENDPOINTS:
        endpoint_data = (
            evidence_long.loc[
                evidence_long["endpoint"].eq(endpoint),
                [
                    SEQUENCE_COL,
                    "status",
                    "is_ambiguous",
                ],
            ]
            .set_index(SEQUENCE_COL)
            .reindex(sequences)
        )

        status_is_ambiguous = (
            endpoint_data["status"].eq("ambiguous")
        )
        ambiguity_flag = (
            endpoint_data["is_ambiguous"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )

        if not status_is_ambiguous.equals(
            ambiguity_flag
        ):
            raise AssertionError(
                f"{endpoint}: status and ambiguity flag "
                "are not synchronized."
            )

        wide[endpoint] = encode_endpoint_state(
            endpoint_data["status"]
        )

    wide = wide.reset_index()

    wide.insert(
        0,
        "id",
        wide[SEQUENCE_COL].map(
            generate_sequence_id
        ),
    )

    return wide


def positive_support_category(value) -> str:
    """
    Convert the positive-evidence fraction to the original 10-point interval.

    Examples
    --------
    0.27 -> 20-30
    0.75 -> 70-80
    """
    if pd.isna(value):
        return "999"

    percentage = float(value) * 100.0
    lower = int(np.floor(percentage / 10.0) * 10)
    lower = max(0, min(lower, 90))
    upper = lower + 10

    return f"{lower}-{upper}"


def build_ambiguous_support_file(
    evidence_long: pd.DataFrame,
):
    """
    Build the separate ambiguous-sequence support table.

    It contains only sequences ambiguous in at least one of these endpoints:
    cytotoxic, hemolytic, neurotoxic, and toxic.

    Values such as 20-30 or 70-80 indicate the positive-support interval.
    The value 999 means no ambiguous information for that endpoint.
    """
    ambiguous_endpoints = [
        "cytotoxic",
        "hemolytic",
        "neurotoxic",
        "toxic",
    ]

    relevant = evidence_long.loc[
        evidence_long["endpoint"].isin(ambiguous_endpoints),
        [
            SEQUENCE_COL,
            "endpoint",
            "is_ambiguous",
            "positive_evidence_fraction",
        ],
    ].copy()

    ambiguous_sequences = (
        relevant.loc[
            relevant["is_ambiguous"],
            SEQUENCE_COL,
        ]
        .drop_duplicates()
        .sort_values()
    )

    result = pd.DataFrame(
        {SEQUENCE_COL: ambiguous_sequences.to_numpy()}
    )

    for endpoint in ambiguous_endpoints:
        endpoint_data = (
            relevant.loc[
                relevant["endpoint"].eq(endpoint),
                [
                    SEQUENCE_COL,
                    "is_ambiguous",
                    "positive_evidence_fraction",
                ],
            ]
            .set_index(SEQUENCE_COL)
            .reindex(ambiguous_sequences)
        )

        endpoint_ambiguous = (
            endpoint_data["is_ambiguous"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )

        result[endpoint] = [
            positive_support_category(value)
            if ambiguous
            else 999
            for ambiguous, value in zip(
                endpoint_ambiguous,
                endpoint_data[
                    "positive_evidence_fraction"
                ],
            )
        ]

    result.insert(
        0,
        "id",
        result[SEQUENCE_COL].map(
            generate_sequence_id
        ),
    )

    return result


def endpoint_summary(
    evidence_long: pd.DataFrame,
):
    """
    Official endpoint summary.

    The four displayed categories are mutually exclusive and correspond
    exactly to the codes exported in maomao_sequence_pivot.csv.
    """
    status_counts = (
        evidence_long
        .groupby(["endpoint", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=[
                "positive",
                "negative",
                "ambiguous",
                "unlabeled",
            ],
            fill_value=0,
        )
        .astype(int)
        .reset_index()
        .rename(
            columns={
                "endpoint": "toxicity_endpoint"
            }
        )
    )

    return status_counts[
        [
            "toxicity_endpoint",
            "positive",
            "negative",
            "ambiguous",
            "unlabeled",
        ]
    ]


def endpoint_processing_audit(evidence_long: pd.DataFrame):
    """
    Technical endpoint-level audit.

    These columns explain how the canonical states were constructed, but they
    are intentionally kept outside the official endpoint-count table.
    """
    audit = (
        evidence_long
        .groupby("endpoint")
        .agg(
            n_sequences=(SEQUENCE_COL, "size"),
            n_direct_positive=("has_direct_positive", "sum"),
            n_direct_negative=("has_direct_negative", "sum"),
            n_direct_ambiguous=("has_direct_ambiguous", "sum"),
            n_direct_unlabeled=("has_direct_unlabeled", "sum"),
            n_sequences_with_ambiguous_evidence=("is_ambiguous", "sum"),
            n_positive_added_by_hierarchy=("is_hierarchy_inferred", "sum"),
            n_positive_hierarchy_conflicts=(
                "has_hierarchical_conflict",
                "sum",
            ),
            n_hierarchy_blocked_by_ambiguity=(
                "hierarchy_blocked_by_ambiguity",
                "sum",
            ),
            n_status_file_mismatches=("status_file_mismatch", "sum"),
            n_status_membership_conflicts=(
                "has_status_membership_conflict",
                "sum",
            ),
        )
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"endpoint": "toxicity_endpoint"})
    )

    return audit


def run_assertions(
    evidence_long: pd.DataFrame,
    wide: pd.DataFrame,
    hierarchy_audit: pd.DataFrame,
):
    checks = []

    def add(name, passed, detail=""):
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "detail": str(detail),
            }
        )

    add(
        "one row per sequence-endpoint",
        not evidence_long.duplicated(
            [SEQUENCE_COL, "endpoint"]
        ).any(),
    )

    add(
        "one row per sequence in wide pivot",
        wide[SEQUENCE_COL].is_unique,
    )

    # Verify the sequence-derived SHA-256 identifiers.
    expected_ids = wide[SEQUENCE_COL].map(
        generate_sequence_id
    )

    add(
        "one unique SHA-256 identifier per sequence",
        wide["id"].is_unique,
    )

    add(
        "sequence identifiers match SHA-256 digests",
        wide["id"].equals(expected_ids),
    )

    ambiguity_matches_status = (
        evidence_long["is_ambiguous"]
        .astype(bool)
        .eq(
            evidence_long["status"].eq(
                "ambiguous"
            )
        )
        .all()
    )

    add(
        "ambiguity flag equals final ambiguous status",
        ambiguity_matches_status,
    )

    add(
        "no positive-and-ambiguous final overlap",
        not (
            evidence_long["status"].eq("positive")
            & evidence_long["is_ambiguous"]
        ).any(),
    )

    support = evidence_long[
        "positive_evidence_fraction"
    ].dropna()

    add(
        "positive evidence fraction within [0,1]",
        support.between(0, 1).all(),
    )

    add(
        "hierarchy audit only contains ontology parents",
        hierarchy_audit["endpoint"]
        .isin(PARENT_CHILDREN)
        .all(),
    )

    applied = hierarchy_audit[
        "is_hierarchy_inferred"
    ]

    add(
        "applied hierarchy yields positive status",
        hierarchy_audit.loc[
            applied,
            "status",
        ].eq("positive").all(),
    )

    blocked = hierarchy_audit[
        "hierarchy_blocked_by_ambiguity"
    ]

    add(
        "ambiguous parents remain ambiguous",
        hierarchy_audit.loc[
            blocked,
            "status",
        ].eq("ambiguous").all(),
    )

    add(
        "cytolysis is never hierarchy-inferred",
        not evidence_long.loc[
            evidence_long["endpoint"].eq(
                "cytolysis"
            ),
            "is_hierarchy_inferred",
        ].any(),
    )

    valid_codes = {0, 1, 2, 3, 999}

    code_by_status = {
        "negative": 0,
        "positive": 1,
        "ambiguous": 2,
        "unlabeled": 3,
        "no_information": 999,
    }

    for endpoint in ENDPOINTS:
        endpoint_rows = evidence_long.loc[
            evidence_long["endpoint"].eq(
                endpoint
            )
        ]

        endpoint_codes = set(
            wide[endpoint]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        add(
            f"{endpoint}: valid compact codes",
            endpoint_codes.issubset(
                valid_codes
            ),
            f"observed={sorted(endpoint_codes)}",
        )

        for status_name, code in (
            code_by_status.items()
        ):
            expected = int(
                endpoint_rows["status"]
                .eq(status_name)
                .sum()
            )

            observed = int(
                wide[endpoint]
                .eq(code)
                .sum()
            )

            add(
                (
                    f"{endpoint}: {status_name} "
                    "summary matches pivot"
                ),
                observed == expected,
                (
                    f"expected={expected}; "
                    f"observed={observed}"
                ),
            )

    return pd.DataFrame(checks)

def build_all(cfg: Config):
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    source_raw, row_raw, manifest, source_column_report = collect_inputs(cfg)
    source_resolved, source_conflicts = resolve_source_evidence(source_raw)
    direct = build_direct_evidence(row_raw, source_resolved)
    complete = complete_grid(direct, cfg)
    evidence_long, hierarchy_audit = apply_positive_only_hierarchy(complete)
    wide = build_wide_pivot(evidence_long)
    ambiguous_support = build_ambiguous_support_file(
        evidence_long
    )
    summary = endpoint_summary(evidence_long)
    processing_audit = endpoint_processing_audit(evidence_long)
    assertions = run_assertions(evidence_long, wide, hierarchy_audit)

    mismatch_audit = row_raw[row_raw["status_file_mismatch"]].copy()
    membership_conflicts = direct[direct["has_status_membership_conflict"]].copy()

    outputs = {
        "maomao_sequence_pivot.csv": wide,
        "maomao_ambiguous_support.csv": ambiguous_support,
        "audit_endpoint_counts.csv": summary,
        "audit_hierarchy_changes.csv": hierarchy_audit,
    }
    for filename, dataframe in outputs.items():
        dataframe.to_csv(cfg.output_root / filename, index=False)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(cfg.input_root),
        "output_root": str(cfg.output_root),
        "hierarchy_rule": "positive support propagates upward, but an ambiguous parent remains ambiguous",
        "ambiguity_rule": "endpoint-specific ambiguity has priority over hierarchical positivity and is the final canonical state",
        "row_status_rule": "stage-2 internal flags take priority over status filename",
        "n_unique_sequences": int(wide.shape[0]),
        "n_sequence_endpoint_rows": int(evidence_long.shape[0]),
        "main_pivot_encoding": {
            "0": "negative",
            "1": "positive",
            "2": "ambiguous",
            "3": "unlabeled",
            "999": "no information",
        },
        "main_pivot_priority": (
            "ambiguous > hierarchical positive; final states are mutually exclusive"
        ),
        "summary_rule": (
            "audit_endpoint_counts.csv is calculated from the same final "
            "exclusive status exported to maomao_sequence_pivot.csv"
        ),
        "assertions_passed": bool(assertions["passed"].all()),
    }

    return {
        "evidence_long": evidence_long,
        "wide": wide,
        "ambiguous_support": ambiguous_support,
        "summary": summary,
        "processing_audit": processing_audit,
        "hierarchy_audit": hierarchy_audit,
        "status_file_mismatches": mismatch_audit,
        "status_membership_conflicts": membership_conflicts,
        "source_conflicts": source_conflicts,
        "manifest": manifest,
        "assertions": assertions,
        "metadata": metadata,
    }
