import json
import pandas as pd
from pathlib import Path

def load_data_if_nonempty(key, activity, filename, path):
    """
    Load a CSV file only if the corresponding metadata entry indicates
    that the dataset is non-empty.

    This function reads a metadata JSON file containing dataset statistics
    and decides whether a specific CSV file should be loaded based on
    the value associated with a given key.

    Parameters
    ----------
    key : str
        Key used to access the relevant statistics entry in the metadata
        file (e.g. "only_unlabel", "ambiguous", "positive", "negative").
    activity : str
        Therapeutic activity name used to locate the dataset directory.
    filename : str
        Name of the CSV file to load if the dataset is non-empty.

    Returns
    -------
    pd.DataFrame or None
        Loaded DataFrame if the dataset contains at least one sequence.
        Returns None if the dataset is empty or the metadata entry
        is missing.
    """

    # Load metadata from JSON
    with open(f"{path}/{activity}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Extract statistics section
    stats = metadata.get("statistics", {})
    entry = stats.get(key)

    # If the requested key does not exist, do not load anything
    if entry is None:
        return None

    # Case 1: entry is a direct integer
    if isinstance(entry, int):
        if entry > 0:
            return pd.read_csv(
                f"{path}/{activity}/{filename}"
            )
        return None

    # Case 2: entry is a dictionary with "n_sequences"
    if isinstance(entry, dict):
        if "n_sequences" in entry and entry["n_sequences"] > 0:
            return pd.read_csv(
                f"{path}/{activity}/{filename}"
            )

        # Case 3: entry is a dictionary with multiple numeric values
        numeric_values = [
            v for v in entry.values() if isinstance(v, (int, float))
        ]

        if sum(numeric_values) > 0:
            return pd.read_csv(
                f"{path}/{activity}/{filename}"
            )

    # Dataset is empty or invalid according to metadata
    return None

def read_all_toxic(base_dir, filename, recursive=True):
    """
    Load and concatenate CSV files with the same name from subdirectories.

    Parameters
    ----------
    base_dir : str or Path
        Base directory to search in.
    filename : str
        Name of the CSV file to look for (e.g. 'positive.csv').
    recursive : bool, optional
        If True, search recursively in subdirectories.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame.
    """
    base_dir = Path(base_dir)

    pattern = f"**/{filename}" if recursive else filename

    dfs = []
    for f in base_dir.glob(pattern):
        df = pd.read_csv(f, usecols=["sequence"])
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No files named '{filename}' were found in {base_dir}"
        )

    return pd.concat(dfs, ignore_index=True)

def load_positive_negative(effect_toxic, base_dir):
    """
    Load positive and negative sequence datasets depending on the toxicity effect.

    Parameters
    ----------
    effect_toxic : str
        Type of toxicity effect. If equal to "toxic", data are loaded from
        multiple subdirectories using `read_all_toxic`. Otherwise, data are
        loaded from a fixed post-processing directory.
    base_dir : str or Path
        Base directory used when effect_toxic is "toxic".

    Returns
    -------
    df_positive : pd.DataFrame
        DataFrame containing positive sequences.
    df_negative : pd.DataFrame
        DataFrame containing negative sequences.
    """

    # Case 1: all toxic effect
    if effect_toxic == "toxic":
        df_positive = read_all_toxic(f"{base_dir}/dataset_post_processing", "positive.csv")
        df_negative = read_all_toxic(f"{base_dir}/dataset_post_processing", "negative.csv")

    # Case 2: Specific toxic effect
    else:
        df_positive = load_data_if_nonempty("positive", effect_toxic, "positive.csv")
        df_negative = load_data_if_nonempty("negative", effect_toxic, "negative.csv")

        df_positive = (
            df_positive[["sequence"]]
            if df_positive is not None
            else pd.DataFrame(columns=["sequence"])
        )

        df_negative = (
            df_negative[["sequence"]]
            if df_negative is not None
            else pd.DataFrame(columns=["sequence"])
        )

    return df_positive, df_negative



def load_labeled_sequences(activity):
    """
    Load positive and negative sequence datasets for a given activity,
    add labels, concatenate them, and compute sequence length.

    Parameters
    ----------
    activity : str

    Returns
    -------
    pd.DataFrame or None
    """

    dfs = []

    # Positive
    df_positive = load_data_if_nonempty(
        key="positive",
        activity=activity,
        filename="positive.csv"
    )
    if df_positive is not None and not df_positive.empty:
        df_positive = df_positive.iloc[:, [0]].assign(label="Positive")
        dfs.append(df_positive)

    # Negative
    df_negative = load_data_if_nonempty(
        key="negative",
        activity=activity,
        filename="negative.csv"
    )
    if df_negative is not None and not df_negative.empty:
        df_negative = df_negative.iloc[:, [0]].assign(label="Negative")
        dfs.append(df_negative)

    if not dfs:
        return None

    df_data = pd.concat(dfs, ignore_index=True)
    return df_data
