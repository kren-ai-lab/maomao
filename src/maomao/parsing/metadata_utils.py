from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json, os, sys, platform
from typing import Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import sklearn


def build_dataset_metadata(
    task: str,
    source_list: list,
    pivote_df: pd.DataFrame,
    outputs: dict,
    seq_stats: dict,
    filters: dict | None = None,
    extra: dict | None = None
) -> dict:
    """
    Build a standardized metadata dictionary for datasets

    This function collects:
    - provenance information (task name, generation timestamp, sources),
    - applied preprocessing filters (e.g., canonical residues, length filter),
    - global sequence statistics before/after filtering,
    - final dataset statistics (positive/negative/unlabeled/ambiguous splits),
    - optional extra fields for downstream tracking.

    Parameters
    ----------
    task : str
        Identifier describing the dataset generation task.
    source_list : list
        List of (source_name, df) pairs used to build the dataset.
        Each df must contain a "sequence" column.
    pivote_df : pd.DataFrame
        Pivot/aggregation dataframe containing boolean columns describing
        class membership (e.g., "positive", "negative", "only_unlabel").
    outputs : dict
        Dictionary containing output dataframes (e.g., "only_positive",
        "only_negative", and optionally "ambiguous").
    seq_stats : dict
        Dictionary with sequence filtering statistics, expected to include:
        - seq_stats["canonical"]["before"], ["after"]
        - seq_stats["length"]["before"], ["after"], ["min"], ["max"]
        - seq_stats["length_dist"]["min"], ["max"], ["mean"], ["median"]
    filters : dict, optional
        Dictionary describing which filters were applied (e.g.,
        {"canonical_residues": True, "length_filter": True}).
    extra : dict, optional
        Extra arbitrary metadata fields to attach.

    Returns
    -------
    dict
        Metadata dictionary that can be serialized to JSON.
    """

    # Initialize the metadata structure
    metadata = {
        "task": task,
        "generated_at": datetime.now().isoformat(),
        "sources": {},
        "filters": {},
        "sequence_statistics": {},
        "statistics": {}
    }

    # Sources: unique sequences per input source
    metadata["sources"]["n_unique_sequences"] = {}
    for source_name, df in source_list:
        metadata["sources"]["n_unique_sequences"][source_name] = int(
            df["sequence"].nunique()
        )

    # Filters: record which preprocessing steps were applied
    if filters:
        if filters.get("canonical_residues"):
            metadata["filters"]["canonical_residues"] = {"applied": True}

        if filters.get("length_filter"):
            metadata["filters"]["length_filter"] = {
                "applied": True,
                "min_length": int(seq_stats["length"]["min"]),
                "max_length": int(seq_stats["length"]["max"])
            }

    # Sequence statistics: counts before/after each filter + length distribution
    metadata["sequence_statistics"] = {
        "canonical_filter": {
            "before": int(seq_stats["canonical"]["before"]),
            "after": int(seq_stats["canonical"]["after"])
        },
        "length_filter": {
            "before": int(seq_stats["length"]["before"]),
            "after": int(seq_stats["length"]["after"])
        },
        "length_distribution": {
            "min": int(seq_stats["length_dist"]["min"]),
            "max": int(seq_stats["length_dist"]["max"]),
            "mean": round(float(seq_stats["length_dist"]["mean"]), 2),
            "median": round(float(seq_stats["length_dist"]["median"]), 2)
        }
    }

    # Final dataset: total sequences after all preprocessing
    metadata["statistics"]["total_sequences_final"] = int(pivote_df.shape[0])

    # Positive / Negative / Unlabeled summary
    # pivote_df is expected to contain boolean columns:
    # "positive", "negative", "only_unlabel"
    positive_and_unlabel = pivote_df[pivote_df["positive"]]
    negative_and_unlabel = pivote_df[pivote_df["negative"]]
    only_unlabel = pivote_df[pivote_df["only_unlabel"]]

    metadata["statistics"]["positive"] = {
        "positive_and_unlabel": int(positive_and_unlabel.shape[0]),
        "only_positive": int(outputs["only_positive"].shape[0])
    }

    metadata["statistics"]["negative"] = {
        "negative_and_unlabel": int(negative_and_unlabel.shape[0]),
        "only_negative": int(outputs["only_negative"].shape[0])
    }

    metadata["statistics"]["only_unlabel"] = int(only_unlabel.shape[0])

    # Ambiguous sequences
    # outputs may include an "ambiguous" dataframe
    df_ambiguous = outputs.get("ambiguous")

    # NOTE: if df_ambiguous is None, df_ambiguous.shape will raise.
    # This assumes you always provide an empty DataFrame at minimum.
    metadata["statistics"]["ambiguous"] = {
        "n_sequences": int(df_ambiguous.shape[0])
    }

    # If ambiguous data includes Category_pbb, store distribution details
    if (
        isinstance(df_ambiguous, pd.DataFrame)
        and df_ambiguous.shape[0] > 0
        and "Category_pbb" in df_ambiguous.columns
    ):
        metadata["statistics"]["ambiguous"]["category_pbb"] = {
            "description": (
                "Distribution of ambiguous sequences based on the "
                "percentage of positive annotations across sources"
            ),
            "categories_definition": (
                "Bins represent the percentage of sources labeling a "
                "sequence as positive"
            ),
            "percentage": (
                df_ambiguous["Category_pbb"]
                .value_counts()
                .to_dict()
            )
        }

    # Extra arbitrary metadata (optional)
    if extra:
        metadata["extra"] = extra

    return metadata


def save_toxic_therapeutic_metadata(
    df_toxic: pd.DataFrame,
    df_toxic_positive: pd.DataFrame,
    df_toxic_negative: pd.DataFrame,
    df_therapeutic: pd.DataFrame,
    df_map: pd.DataFrame,
    df_toxic_only_positive: pd.DataFrame,
    df_toxic_only_negative: pd.DataFrame,
    df_toxic_only_all: pd.DataFrame,
    df_therapeutic_only: pd.DataFrame,
    df_therapeutic_toxic_positive: pd.DataFrame,
    df_therapeutic_toxic_negative: pd.DataFrame,
    df_therapeutic_toxic_all: pd.DataFrame,
    counts_global: pd.DataFrame | None = None,
    counts_by_activity: pd.DataFrame | None = None,
    counts_by_effect: pd.DataFrame | None = None,
    effect_activity_pairs: pd.DataFrame | None = None,
):
    """
    Build metadata describing a toxicity–therapeutic sequence mapping dataset.

    This mapping integrates toxic-effect annotations (positive and negative when available)
    and therapeutic activity annotations at sequence level. Sequences are classified into
    detailed and general categories according to toxicity label and therapeutic activity.

    Returns
    -------
    dict
        Metadata dictionary that can be serialized to JSON.
    """

    metadata = {
        "task": "toxicity_therapeutic_mapping",
        "generated_at": datetime.now().isoformat(),

        "source_datasets": {
            "therapeutic_activities": "Peptipedia 2.0",
            "toxic_positive": "../../dataset_post_processing/*/positive.csv",
            "toxic_negative": "../../dataset_post_processing/*/negative.csv",
        },

        "categories_general": {
            "toxic_only": {
                "description": "Sequences mapped to toxic annotations but not to therapeutic activities",
                "logic": "category_detailed in ['toxic_only_positive', 'toxic_only_negative']"
            },
            "therapeutic_only": {
                "description": "Sequences mapped to therapeutic activities and not to toxic annotations",
                "logic": "category_detailed == 'therapeutic_only'"
            },
            "therapeutic_toxic": {
                "description": "Sequences mapped to both toxic annotations and therapeutic activities",
                "logic": "category_detailed in ['therapeutic_toxic_positive', 'therapeutic_toxic_negative']"
            },
        },

        "categories_detailed": {
            "toxic_only_positive": {
                "description": "Sequences with toxic positive label and no therapeutic activity",
                "logic": "is_toxic_positive == true AND is_therapeutic == false"
            },
            "toxic_only_negative": {
                "description": "Sequences with toxic negative label and no therapeutic activity",
                "logic": "is_toxic_negative == true AND is_therapeutic == false"
            },
            "therapeutic_only": {
                "description": "Sequences with therapeutic activity and no toxic annotation",
                "logic": "is_toxic_positive == false AND is_toxic_negative == false AND is_therapeutic == true"
            },
            "therapeutic_toxic_positive": {
                "description": "Sequences with therapeutic activity and toxic positive label",
                "logic": "is_toxic_positive == true AND is_therapeutic == true"
            },
            "therapeutic_toxic_negative": {
                "description": "Sequences with therapeutic activity and toxic negative label",
                "logic": "is_toxic_negative == true AND is_therapeutic == true"
            },
        },

        "label_spaces": {
            "toxic_effects_all": sorted(df_toxic["effect"].dropna().unique().tolist()),
            "toxic_effects_positive": sorted(df_toxic_positive["effect"].dropna().unique().tolist()),
            "toxic_effects_negative": sorted(df_toxic_negative["effect"].dropna().unique().tolist()),
            "therapeutic_activities": sorted(df_therapeutic["activity"].dropna().unique().tolist()),
            "toxic_labels": sorted(df_toxic["label"].dropna().unique().tolist()),
        },

        "statistics": {
            "n_unique_sequences_total": int(df_map["sequence"].nunique()),

            "input_tables": {
                "toxic_rows_total": int(len(df_toxic)),
                "toxic_positive_rows": int(len(df_toxic_positive)),
                "toxic_negative_rows": int(len(df_toxic_negative)),
                "therapeutic_rows_total": int(len(df_therapeutic)),
                "toxic_unique_sequences": int(df_toxic["sequence"].nunique()),
                "therapeutic_unique_sequences": int(df_therapeutic["sequence"].nunique()),
            },

            "global_categories_general": {
                "toxic_only": int(df_toxic_only_all["sequence"].nunique()),
                "therapeutic_only": int(df_therapeutic_only["sequence"].nunique()),
                "therapeutic_toxic": int(df_therapeutic_toxic_all["sequence"].nunique()),
            },

            "global_categories_detailed": {
                "toxic_only_positive": int(df_toxic_only_positive["sequence"].nunique()),
                "toxic_only_negative": int(df_toxic_only_negative["sequence"].nunique()),
                "therapeutic_only": int(df_therapeutic_only["sequence"].nunique()),
                "therapeutic_toxic_positive": int(df_therapeutic_toxic_positive["sequence"].nunique()),
                "therapeutic_toxic_negative": int(df_therapeutic_toxic_negative["sequence"].nunique()),
            },
        }
    }

    return metadata

def register_reduction(metadata, method, params, n_before, n_after):
    """
    Append a single redundancy-reduction step to an in-memory list.

    Parameters
    ----------
    metadata : list
        List that stores reduction steps as dictionaries.
    method : str
        Reduction method name (e.g. "CD-HIT", "MMseqs2", "dedup").
    params : dict
        Parameters used by the reduction method (e.g. identity threshold).
    n_before : int
        Number of sequences before reduction.
    n_after : int
        Number of sequences after reduction.

    Returns
    -------
    None
        The metadata list is modified in place.
    """
    metadata.append({
        "method": method,
        "parameters": params,
        "n_before": n_before,
        "n_after": n_after,
    })

def build_redundancy_metadata(
    *,
    task: str,
    effect: str,
    source_datasets: dict,
    reductions: list,
):
    """
    Build a metadata dictionary summarizing redundancy reduction operations.

    Parameters
    ----------
    task : str
        Identifier describing the redundancy-reduction task.
    effect : str
        Name of the toxic effect or label being processed.
    source_datasets : dict
        Description of the input datasets used.
    reductions : list
        List of reduction steps (typically created via register_reduction).

    Returns
    -------
    dict
        Metadata dictionary that can be serialized to JSON.
    """

    return {
        "task": task,
        "generated_at": datetime.utcnow().isoformat(),
        "effect": effect,
        "source_datasets": source_datasets,
        "reductions": reductions,
    }


def _file_info(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime_utc": datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
    }


def _try_get_params(obj: Any) -> Optional[dict]:
    if hasattr(obj, "get_params"):
        try:
            return obj.get_params(deep=False)
        except Exception:
            return None
    return None


# Metadata hierarchical
MISSING_VALUE = 999

def value_counts_dict(series):
    """
    Return value counts as a JSON-serializable dictionary.
    """
    return {
        str(k): int(v)
        for k, v in series.value_counts(dropna=False).sort_index().items()
    }

def dataframe_records(df):
    """
    Convert a dataframe into JSON-serializable records.
    """
    return json.loads(df.to_json(orient="records"))


def prepare_metadata_long_table(df_all):
    """
    Validate and prepare the long-format annotation table used for metadata.

    Expected columns:
    - sequence
    - task
    - label
    """
    required_columns = {"sequence", "task", "label"}
    missing_columns = required_columns.difference(df_all.columns)

    if missing_columns:
        raise ValueError(
            f"df_all must contain the following columns to build metadata: "
            f"{required_columns}. Missing columns: {missing_columns}"
        )

    df_all_metadata = df_all[["sequence", "task", "label"]].copy()
    df_all_metadata["sequence"] = df_all_metadata["sequence"].astype(str)
    df_all_metadata["task"] = df_all_metadata["task"].astype(str)
    df_all_metadata["label"] = df_all_metadata["label"].astype(int)

    return df_all_metadata

def long_annotation_summary(df_long):
    """
    Summarize the original long-format annotation table.

    This is used for df_all before pivoting.
    In this table, each row corresponds to an observed annotation.
    """
    task_summary = (
        df_long
        .groupby("task")
        .agg(
            n_records=("sequence", "size"),
            n_unique_sequences=("sequence", "nunique"),
            n_positive=("label", lambda x: int((x == 1).sum())),
            n_negative=("label", lambda x: int((x == 0).sum()))
        )
        .reset_index()
    )

    #task_summary["n_missing"] = 0

    return {
        "n_unique_sequences": int(df_long["sequence"].nunique()),
        "n_tasks": int(df_long["task"].nunique()),
        "tasks": sorted(df_long["task"].unique().tolist()),
        "task_summary": dataframe_records(task_summary)
    }

def rule_change_report(
    rule_id,
    description,
    source_terms,
    target_term,
    trigger_mask,
    before_df,
    after_df,
    relation,
    condition,
    action,
    overwrites_negative=True,
    missing_value=999
):
    """Summarize the effect of a hierarchy rule without changing the original notebook logic."""
    before = before_df[target_term]
    after = after_df[target_term]

    changed = before.ne(after)
    changed_to_positive = changed & after.eq(1)

    return {
        "rule_id": rule_id,
        "description": description,
        "source_terms": source_terms,
        "target_term": target_term,
        "relation": relation,
        "condition": condition,
        "action": action,
        "overwrites_negative": bool(overwrites_negative),
        "n_triggered_sequences": int(trigger_mask.sum()),
        "n_target_changed": int(changed.sum()),
        "n_target_changed_to_positive": int(changed_to_positive.sum()),
        "n_changed_from_missing_to_positive": int(
            (changed_to_positive & before.eq(missing_value)).sum()
        ),
        "n_changed_from_negative_to_positive": int(
            (changed_to_positive & before.eq(0)).sum()
        ),
        "n_already_positive_among_triggered": int(
            (trigger_mask & before.eq(1)).sum()
        ),
        "target_label_counts_before": value_counts_dict(before),
        "target_label_counts_after": value_counts_dict(after)
    }

def wide_annotation_summary(df_wide, sequence_col="sequence", missing_value=999):
    """
    Summarize a wide-format annotation table.

    This is used for df_pivot before and after hierarchy correction.
    In this table, each row corresponds to one unique sequence.
    """
    annotation_cols = [
        col for col in df_wide.columns
        if col != sequence_col
    ]

    task_summary = []

    for col in annotation_cols:
        n_positive = int((df_wide[col] == 1).sum())
        n_negative = int((df_wide[col] == 0).sum())
        n_missing = int((df_wide[col] == missing_value).sum())

        # Real annotations are positive or negative.
        # Missing values encoded as 999 are not counted as records.
        n_records = n_positive + n_negative

        task_summary.append({
            "task": col,
            "n_records": int(n_records),
            "n_unique_sequences": int(n_records),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_missing": n_missing,
            "label_distribution": value_counts_dict(df_wide[col])
        })

    return {
        "n_unique_sequences": int(df_wide[sequence_col].nunique()),
        "n_tasks": int(len(annotation_cols)),
        "tasks": sorted(annotation_cols),
        "task_summary": task_summary
    }

def positive_only_rule_report(
    rule_id,
    description,
    source_terms,
    target_term,
    trigger_mask,
    before_df,
    after_df,
    relation,
    condition,
    action
):
    """
    Summarize the effect of a positive-only hierarchy rule.

    Important:
    - Positive labels can be propagated upward in the hierarchy.
    - Negative labels are NOT propagated by hierarchy.
    - Negative labels remain direct endpoint-specific annotations.
    """

    before = before_df[target_term]
    after = after_df[target_term]

    changed = before.ne(after)
    changed_to_positive = changed & after.eq(1)

    return {
        "rule_id": rule_id,
        "description": description,
        "source_terms": source_terms,
        "target_term": target_term,
        "relation": relation,
        "condition": condition,
        "action": action,

        "propagation_policy": "positive_only",
        "positive_labels_propagated_by_hierarchy": True,
        "negative_labels_propagated_by_hierarchy": False,

        "negative_label_interpretation": (
            "Negative labels are retained only as direct endpoint-specific "
            "annotations from the original source files. They are not inferred, "
            "propagated, or expanded through the hierarchy."
        )
    }

def reconstruct_pre_hierarchy_pivot(df_all_metadata, missing_value=MISSING_VALUE):
    """Reconstruct the pre-hierarchy pivot without modifying the notebook df_pivot."""
    df_pivot_before_hierarchy = (
        df_all_metadata.pivot_table(
            index="sequence",
            columns="task",
            values="label",
            aggfunc="first"
        )
        .reset_index()
    )

    df_pivot_before_hierarchy.columns.name = None
    df_pivot_before_hierarchy = df_pivot_before_hierarchy.fillna(missing_value)

    for column in df_pivot_before_hierarchy.columns:
        if column != "sequence":
            df_pivot_before_hierarchy[column] = df_pivot_before_hierarchy[column].astype(int)

    return df_pivot_before_hierarchy


def controlled_vocabulary_metadata():
    """Return the local controlled vocabulary and hierarchy metadata."""
    return {
        "version": "1.0",
        "description": (
            "Controlled vocabulary and hierarchy used to harmonize peptide toxicity "
            "annotations across heterogeneous sources."
        ),
        "hierarchy_policy": {
            "positive_propagation": True,
            "negative_propagation": False,
        },
        "terms": [
            {
                "term": "toxic",
                "label_type": "broad toxicity endpoint",
                "definition": "General evidence of peptide toxicity."
            },
            {
                "term": "cytotoxic",
                "label_type": "intermediate toxicity endpoint",
                "definition": "Evidence of toxicity affecting cells or cell viability.",
                "parent_terms": ["toxic"]
            },
            {
                "term": "cytolysis",
                "label_type": "specific toxicity endpoint",
                "definition": "Evidence of cell lysis or membrane-disruptive toxicity.",
                "parent_terms": ["cytotoxic", "toxic"]
            },
            {
                "term": "cytolytic",
                "label_type": "source-specific term",
                "definition": "Source-specific cytolytic annotation harmonized with cytolysis.",
                "harmonized_to": "cytolysis"
            },
            {
                "term": "hemolytic",
                "label_type": "specific toxicity endpoint",
                "definition": "Evidence of erythrocyte lysis or hemolytic activity.",
                "parent_terms": ["cytotoxic", "toxic"]
            },
            {
                "term": "embryotoxic",
                "label_type": "specific toxicity endpoint",
                "definition": "Evidence of toxicity affecting embryos or embryonic development.",
                "parent_terms": ["toxic"]
            },
            {
                "term": "ichthyotoxic",
                "label_type": "specific toxicity endpoint",
                "definition": "Evidence of toxicity affecting fish.",
                "parent_terms": ["toxic"]
            },
            {
                "term": "neurotoxic",
                "label_type": "specific toxicity endpoint",
                "definition": "Evidence of toxicity affecting the nervous system.",
                "parent_terms": ["toxic"]
            }
        ]
    }


def positive_only_rule_report(
    rule_id,
    description,
    source_terms,
    target_term,
    trigger_mask,
    before_df,
    after_df,
    relation,
    condition,
    action,
    missing_value=MISSING_VALUE
):
    """
    Summarize the effect of a positive-only hierarchy rule.

    Important:
    - Positive labels can be propagated upward in the hierarchy.
    - Negative labels are NOT propagated by hierarchy.
    - Negative labels remain direct endpoint-specific annotations.
    """
    before = before_df[target_term]
    after = after_df[target_term]

    changed = before.ne(after)
    changed_to_positive = changed & after.eq(1)

    return {
        "rule_id": rule_id,
        "description": description,
        "source_terms": source_terms,
        "target_term": target_term,
        "relation": relation,
        "condition": condition,
        "action": action,
        "propagation_policy": "positive_only",
        "positive_labels_propagated_by_hierarchy": True,
        "negative_labels_propagated_by_hierarchy": False,
        "negative_label_interpretation": (
            "Negative labels are retained only as direct endpoint-specific "
            "annotations from the original source files. They are not inferred, "
            "propagated, or expanded through the hierarchy."
        )
    }


def simulate_hierarchy_for_metadata(df_pivot_before_hierarchy, missing_value=MISSING_VALUE):
    """
    Simulate the same hierarchy rules used in the notebook only for metadata reporting.

    This function does not modify the notebook's final df_pivot.
    """
    df_hierarchy_simulated = df_pivot_before_hierarchy.copy(deep=True)
    hierarchy_reports = []

    specific_activity_cols = [
        "cytolysis", "cytolytic", "cytotoxic",
        "embryotoxic", "hemolytic", "ichthyotoxic",
        "neurotoxic"
    ]
    available_specific_activity_cols = [col for col in specific_activity_cols if col in df_hierarchy_simulated.columns]

    before_r1 = df_hierarchy_simulated.copy(deep=True)
    mask_r1 = (
        df_hierarchy_simulated[available_specific_activity_cols]
        .replace(missing_value, pd.NA)
        .eq(1)
        .any(axis=1)
    )
    df_hierarchy_simulated.loc[mask_r1, "toxic"] = 1
    after_r1 = df_hierarchy_simulated.copy(deep=True)

    hierarchy_reports.append(
        positive_only_rule_report(
            rule_id="R1",
            description="Endpoint-specific positive toxicity annotations are propagated to the broad toxic label.",
            source_terms=available_specific_activity_cols,
            target_term="toxic",
            trigger_mask=mask_r1,
            before_df=before_r1,
            after_df=after_r1,
            relation="is_a / broader_than",
            condition="if any endpoint-specific toxicity column == 1",
            action="set toxic = 1",
            missing_value=missing_value
        )
    )

    cytotoxic_source_cols = ["cytolysis", "cytolytic", "hemolytic"]
    available_cytotoxic_source_cols = [col for col in cytotoxic_source_cols if col in df_hierarchy_simulated.columns]

    before_r2 = df_hierarchy_simulated.copy(deep=True)
    mask_r2 = (
        df_hierarchy_simulated[available_cytotoxic_source_cols]
        .replace(missing_value, pd.NA)
        .eq(1)
        .any(axis=1)
    )
    df_hierarchy_simulated.loc[mask_r2, "cytotoxic"] = 1
    after_r2 = df_hierarchy_simulated.copy(deep=True)

    hierarchy_reports.append(
        positive_only_rule_report(
            rule_id="R2",
            description="Positive cytolysis, cytolytic, or hemolytic annotations are propagated to cytotoxic.",
            source_terms=available_cytotoxic_source_cols,
            target_term="cytotoxic",
            trigger_mask=mask_r2,
            before_df=before_r2,
            after_df=after_r2,
            relation="is_a / mechanistically_related_to",
            condition="if cytolysis == 1 or cytolytic == 1 or hemolytic == 1",
            action="set cytotoxic = 1",
            missing_value=missing_value
        )
    )

    cytolysis_source_cols = ["cytolysis", "cytolytic"]
    available_cytolysis_source_cols = [col for col in cytolysis_source_cols if col in df_hierarchy_simulated.columns]

    mask_r3_positive = (
        df_hierarchy_simulated[available_cytolysis_source_cols]
        .replace(missing_value, pd.NA)
        .eq(1)
        .any(axis=1)
    )
    mask_r3_any_negative = df_hierarchy_simulated[available_cytolysis_source_cols].eq(0).any(axis=1)
    mask_r3_negative_only = mask_r3_any_negative & ~mask_r3_positive

    df_hierarchy_simulated["Cytolysis_combined"] = mask_r3_positive.astype(int)
    df_hierarchy_simulated["Cytolysis_combined"] = df_hierarchy_simulated["Cytolysis_combined"].replace({0: missing_value})
    df_hierarchy_simulated = df_hierarchy_simulated.drop(columns=available_cytolysis_source_cols)
    df_hierarchy_simulated = df_hierarchy_simulated.rename(columns={"Cytolysis_combined": "cytolysis"})

    hierarchy_reports.append(
        {
            "rule_id": "R3",
            "description": "Cytolysis and cytolytic annotations are merged into a single harmonized cytolysis label.",
            "source_terms": available_cytolysis_source_cols,
            "target_term": "cytolysis",
            "relation": "synonym_or_equivalent_mapping",
            "condition": "if cytolysis == 1 or cytolytic == 1",
            "action": "set merged cytolysis = 1; otherwise set merged cytolysis = 999",
            "harmonization_policy": "positive_only_term_merge",
            "positive_labels_transferred_by_harmonization": True,
            "negative_labels_transferred_by_harmonization": False,
            "negative_label_interpretation": (
                "Negative cytolysis/cytolytic source annotations are not transferred "
                "to the merged cytolysis column under the current notebook logic. "
                "Therefore, a 0 in the final dataset is not created by this merge rule."
            )
        }
    )

    return df_hierarchy_simulated, hierarchy_reports


def validate_simulated_hierarchy_against_current(df_pivot, df_hierarchy_simulated):
    """Validate that the hierarchy simulated for metadata matches the current notebook df_pivot."""
    current_final = df_pivot.copy(deep=True)
    simulated_final = df_hierarchy_simulated.copy(deep=True)

    same_column_set = set(current_final.columns) == set(simulated_final.columns)

    if same_column_set:
        compare_columns = ["sequence"] + sorted([col for col in current_final.columns if col != "sequence"])
        current_compare = current_final[compare_columns].sort_values("sequence").reset_index(drop=True)
        simulated_compare = simulated_final[compare_columns].sort_values("sequence").reset_index(drop=True)
        matches_current_df_pivot = bool(current_compare.equals(simulated_compare))
    else:
        matches_current_df_pivot = False

    return {
        "matches_current_df_pivot": matches_current_df_pivot,
        "same_column_set": bool(same_column_set),
        "n_rows_current_df_pivot": int(current_final.shape[0]),
        "n_columns_current_df_pivot": int(current_final.shape[1]),
        "n_rows_simulated_from_metadata_logic": int(simulated_final.shape[0]),
        "n_columns_simulated_from_metadata_logic": int(simulated_final.shape[1]),
        "columns_current_df_pivot": current_final.columns.tolist(),
        "columns_simulated_from_metadata_logic": simulated_final.columns.tolist()
    }


def infer_input_files(path_data, tasks):
    """Infer positive and negative input file paths from path_data and observed tasks."""
    input_files = {}

    for task in sorted(list(tasks)):
        positive_path = Path(path_data) / task / "positive.csv"
        negative_path = Path(path_data) / task / "negative.csv"

        input_files[task] = {
            "positive_file": {"path": str(positive_path), "exists": bool(positive_path.exists())},
            "negative_file": {"path": str(negative_path), "exists": bool(negative_path.exists())}
        }

    return input_files


def save_metadata_json(metadata, metadata_output_path):
    """Save metadata as a JSON file."""
    metadata_output_path = Path(metadata_output_path)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return metadata_output_path


def prepare_metadata_context(df_all, df_pivot, path_data, missing_value=MISSING_VALUE):
    """
    Prepare all computed sections required to build the metadata dictionary.

    This keeps the notebook focused on the explicit metadata dictionary structure.
    """
    df_all_metadata = prepare_metadata_long_table(df_all)
    df_pivot_before_hierarchy = reconstruct_pre_hierarchy_pivot(
        df_all_metadata,
        missing_value=missing_value
    )
    df_hierarchy_simulated, hierarchy_reports = simulate_hierarchy_for_metadata(
        df_pivot_before_hierarchy,
        missing_value=missing_value
    )
    validation_report = validate_simulated_hierarchy_against_current(
        df_pivot,
        df_hierarchy_simulated
    )

    final_annotation_columns = [
        col for col in df_pivot.columns
        if col != "sequence"
    ]

    return {
        "input_files": infer_input_files(path_data, df_all_metadata["task"].unique()),
        "controlled_vocabulary_and_hierarchy": controlled_vocabulary_metadata(),
        "raw_annotation_before_hierarchy": long_annotation_summary(df_all_metadata),
        "annotation_before_hierarchy": wide_annotation_summary(
            df_pivot_before_hierarchy,
            sequence_col="sequence",
            missing_value=missing_value
        ),
        "hierarchy_reports": hierarchy_reports,
        "annotation_after_hierarchy": wide_annotation_summary(
            df_pivot,
            sequence_col="sequence",
            missing_value=missing_value
        ),
        "validation": validation_report,
        "final_annotation_columns": final_annotation_columns
    }