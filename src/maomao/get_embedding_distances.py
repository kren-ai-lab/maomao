from __future__ import annotations

from pathlib import Path
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist


def load_label_data(label_path: str | Path) -> pd.DataFrame:
    """
    Load the label table used for merging with numerical representations.

    Parameters
    ----------
    label_path : str or Path
        Path to the labeled dataset. It must contain at least the
        columns 'sequence' and 'label'.

    Returns
    -------
    pd.DataFrame
        Label dataframe.
    """
    return pd.read_csv(label_path)


def load_and_merge_representation(
    representation_path: str | Path,
    df_labels: pd.DataFrame,
    merge_on: str = "sequence",
    how: str = "inner",
) -> pd.DataFrame:
    """
    Load one numerical representation file and merge it with labels.

    Parameters
    ----------
    representation_path : str or Path
        Path to the numerical representation CSV.
    df_labels : pd.DataFrame
        Dataframe with sequence labels.
    merge_on : str, default="sequence"
        Column used for merging.
    how : str, default="inner"
        Merge strategy.

    Returns
    -------
    pd.DataFrame
        Merged dataframe containing embeddings and labels.
    """
    df_rep = pd.read_csv(representation_path)
    df_merged = pd.merge(df_rep, df_labels, on=merge_on, how=how)
    return df_merged


def split_features_and_metadata(
    df_merged: pd.DataFrame,
    feature_drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split merged data into feature matrix and metadata table.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Merged dataframe with embeddings and metadata.
    feature_drop_cols : list[str] or None, default=None
        Columns to exclude from the feature matrix.
        If None, uses ['sequence', 'label'].

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Features dataframe
        - Metadata dataframe with preserved non-feature columns
    """
    if feature_drop_cols is None:
        feature_drop_cols = ["sequence", "label"]

    feature_drop_cols = [c for c in feature_drop_cols if c in df_merged.columns]

    df_features = df_merged.drop(columns=feature_drop_cols)
    df_metadata = df_merged[[c for c in feature_drop_cols if c in df_merged.columns]].copy()

    return df_features, df_metadata


def l2_normalize_features(df_features: pd.DataFrame) -> np.ndarray:
    """
    Apply L2 normalization row-wise to the feature matrix.

    Parameters
    ----------
    df_features : pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        L2-normalized matrix as float32.
    """
    X = df_features.to_numpy(dtype=np.float32)
    X_norm = normalize(X, norm="l2")
    return X_norm.astype(np.float32, copy=False)


def summarize_normalization(X_norm: np.ndarray, n_examples: int = 3) -> pd.DataFrame:
    """
    Build a small table showing example row norms after normalization.

    Parameters
    ----------
    X_norm : np.ndarray
        L2-normalized feature matrix.
    n_examples : int, default=3
        Number of example rows to summarize.

    Returns
    -------
    pd.DataFrame
        Table with row index and L2 norm.
    """
    n_examples = min(n_examples, X_norm.shape[0])

    rows = []
    for i in range(n_examples):
        rows.append({
            "row_index": i,
            "l2_norm": float(np.linalg.norm(X_norm[i]))
        })

    return pd.DataFrame(rows)


def compute_condensed_cosine_distances(X_norm: np.ndarray) -> np.ndarray:
    """
    Compute condensed pairwise cosine distances.

    This avoids building the full NxN similarity matrix and is much more
    memory efficient.

    Parameters
    ----------
    X_norm : np.ndarray
        L2-normalized feature matrix.

    Returns
    -------
    np.ndarray
        Condensed cosine distance vector.
    """
    return pdist(X_norm, metric="cosine").astype(np.float32, copy=False)


def build_pairwise_tables(dist_values: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build pairwise distance and similarity tables from condensed distances.

    Parameters
    ----------
    dist_values : np.ndarray
        Condensed cosine distance vector.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Distance table
        - Similarity table
    """
    sim_values = 1.0 - dist_values

    df_dist = pd.DataFrame({"cosine_distance": dist_values})
    df_sim = pd.DataFrame({"cosine_similarity": sim_values})

    return df_dist, df_sim


def build_summary_table(
    representation_name: str,
    df_merged: pd.DataFrame,
    df_dist: pd.DataFrame,
    df_sim: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact summary table for one numerical representation.

    Parameters
    ----------
    representation_name : str
        Representation name.
    df_merged : pd.DataFrame
        Merged dataframe.
    df_dist : pd.DataFrame
        Distance table.
    df_sim : pd.DataFrame
        Similarity table.

    Returns
    -------
    pd.DataFrame
        One-row summary table.
    """
    out = {
        "representation": representation_name,
        "n_sequences": int(df_merged.shape[0]),
        "n_features": int(df_merged.shape[1] - 2),  # assumes sequence + label
        "n_pairwise_comparisons": int(len(df_dist)),
        "distance_mean": float(df_dist["cosine_distance"].mean()),
        "distance_std": float(df_dist["cosine_distance"].std()),
        "distance_min": float(df_dist["cosine_distance"].min()),
        "distance_q25": float(df_dist["cosine_distance"].quantile(0.25)),
        "distance_median": float(df_dist["cosine_distance"].median()),
        "distance_q75": float(df_dist["cosine_distance"].quantile(0.75)),
        "distance_max": float(df_dist["cosine_distance"].max()),
        "similarity_mean": float(df_sim["cosine_similarity"].mean()),
        "similarity_std": float(df_sim["cosine_similarity"].std()),
        "similarity_min": float(df_sim["cosine_similarity"].min()),
        "similarity_q25": float(df_sim["cosine_similarity"].quantile(0.25)),
        "similarity_median": float(df_sim["cosine_similarity"].median()),
        "similarity_q75": float(df_sim["cosine_similarity"].quantile(0.75)),
        "similarity_max": float(df_sim["cosine_similarity"].max()),
    }

    return pd.DataFrame([out])


def save_representation_outputs(
    output_dir: str | Path,
    representation_name: str,
    df_norm_example: pd.DataFrame,
    df_dist: pd.DataFrame,
    df_sim: pd.DataFrame,
    df_summary: pd.DataFrame,
) -> None:
    """
    Save output tables for one representation.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory.
    representation_name : str
        Name of the numerical representation.
    df_norm_example : pd.DataFrame
        Example normalization summary table.
    df_dist : pd.DataFrame
        Pairwise distance table.
    df_sim : pd.DataFrame
        Pairwise similarity table.
    df_summary : pd.DataFrame
        Summary table.
    """
    rep_dir = Path(output_dir) / representation_name
    rep_dir.mkdir(parents=True, exist_ok=True)

    df_norm_example.to_csv(rep_dir / "normalization_example.csv", index=False)
    df_dist.to_csv(rep_dir / "pairwise_cosine_distance.csv", index=False)
    df_sim.to_csv(rep_dir / "pairwise_cosine_similarity.csv", index=False)
    df_summary.to_csv(rep_dir / "summary.csv", index=False)


def process_one_representation(
    representation_name: str,
    representation_path: str | Path,
    df_labels: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """
    Process one numerical representation end-to-end.

    Steps
    -----
    1. Load representation
    2. Merge with labels
    3. Extract features
    4. L2 normalize
    5. Compute condensed cosine distances
    6. Convert distances to similarity
    7. Save outputs

    Parameters
    ----------
    representation_name : str
        Representation name.
    representation_path : str or Path
        Path to the numerical representation CSV.
    df_labels : pd.DataFrame
        Label dataframe.
    output_dir : str or Path
        Base output folder.

    Returns
    -------
    pd.DataFrame
        One-row summary for this representation.
    """
    print(f"\nProcessing: {representation_name}")

    df_merged = load_and_merge_representation(
        representation_path=representation_path,
        df_labels=df_labels,
        merge_on="sequence",
        how="inner",
    )
    print(f"Merged shape: {df_merged.shape}")

    df_features, _ = split_features_and_metadata(df_merged)
    print(f"Feature matrix shape: {df_features.shape}")

    X_norm = l2_normalize_features(df_features)
    print(f"Normalized shape: {X_norm.shape}")

    df_norm_example = summarize_normalization(X_norm, n_examples=5)

    dist_values = compute_condensed_cosine_distances(X_norm)
    print(f"Number of pairwise comparisons: {len(dist_values)}")

    df_dist, df_sim = build_pairwise_tables(dist_values)
    df_summary = build_summary_table(
        representation_name=representation_name,
        df_merged=df_merged,
        df_dist=df_dist,
        df_sim=df_sim,
    )

    save_representation_outputs(
        output_dir=output_dir,
        representation_name=representation_name,
        df_norm_example=df_norm_example,
        df_dist=df_dist,
        df_sim=df_sim,
        df_summary=df_summary,
    )

    # release memory
    del df_merged, df_features, X_norm, dist_values, df_dist, df_sim
    gc.collect()

    return df_summary


def process_multiple_representations(
    representation_files: dict[str, str],
    label_path: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    """
    Process multiple representations sequentially, one at a time.

    Parameters
    ----------
    representation_files : dict[str, str]
        Mapping {representation_name: csv_path}.
    label_path : str or Path
        Path to labeled data.
    output_dir : str or Path
        Base output directory.

    Returns
    -------
    pd.DataFrame
        Concatenated summary table for all processed representations.
    """
    df_labels = load_label_data(label_path)

    all_summaries = []

    for representation_name, representation_path in representation_files.items():
        summary_df = process_one_representation(
            representation_name=representation_name,
            representation_path=representation_path,
            df_labels=df_labels,
            output_dir=output_dir,
        )
        all_summaries.append(summary_df)

    df_all = pd.concat(all_summaries, ignore_index=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_all.to_csv(Path(output_dir) / "all_representations_summary.csv", index=False)

    return df_all