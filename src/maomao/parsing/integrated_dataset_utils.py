import pandas as pd
from maomao.utils.constants import *

def check_sequence(sequence: str) -> bool:
    """
    Check whether a peptide sequence contains only canonical residues.

    A sequence is considered *canonical* if every character (residue) belongs to
    the predefined set `CANONICAL_RESIDUES` (imported from `commons.constants`).

    Parameters
    ----------
    sequence : str
        Peptide sequence to validate.

    Returns
    -------
    bool
        True if all residues are canonical, False otherwise.

    Notes
    -----
    - The check stops early as soon as a non-canonical residue is detected.
    - This function assumes `sequence` is an iterable of single-letter residues.
    """
    # Assume canonical unless a non-canonical residue appears
    for residue in sequence:
        if residue not in CANONICAL_RESIDUES:
            return False
    return True


def check_length(length_value: int) -> bool:
    """
    Check whether a peptide length lies within an allowed range.

    The valid length range is defined by `MIN_LENGTH_SEQUENCE` and
    `MAX_LENGTH_SEQUENCE` (imported from `commons.constants`).

    Parameters
    ----------
    length_value : int
        Length of the sequence.

    Returns
    -------
    bool
        True if `MIN_LENGTH_SEQUENCE <= length_value <= MAX_LENGTH_SEQUENCE`,
        False otherwise.
    """
    return MIN_LENGTH_SEQUENCE <= length_value <= MAX_LENGTH_SEQUENCE


def count_unique_sequence(df_list: list[pd.DataFrame]) -> list[str]:
    """
    Collect and return unique peptide sequences from multiple DataFrames.

    Parameters
    ----------
    df_list : list[pandas.DataFrame]
        List of DataFrames, each expected to contain a 'sequence' column.

    Returns
    -------
    list[str]
        List of unique sequences across all input DataFrames.

    Side Effects
    ------------
    Prints the number of unique sequences found.

    Notes
    -----
    - Uniqueness is computed by converting to a set.
    - The returned list is not guaranteed to preserve the original ordering.
      If order is important, consider using an ordered-deduplication strategy.
    """
    sequences = []
    for df in df_list:
        # Extend with sequences from current dataframe
        sequences += df["sequence"].values.tolist()

    unique_sequences = list(set(sequences))
    print(len(unique_sequences))
    return unique_sequences


def create_pivote(unique_sequence: list[str]) -> pd.DataFrame:
    """
    Create a pivot-ready DataFrame containing a unique sequence index.

    Parameters
    ----------
    unique_sequence : list[str]
        List of unique peptide sequences.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a single column 'sequence' containing the provided
        sequences, with missing values removed.

    Notes
    -----
    This function provides a standardized "backbone" table onto which label
    columns from different sources can later be merged.
    """
    df_pivoted = pd.DataFrame({"sequence": unique_sequence}).dropna()
    return df_pivoted


def process_count_labels(df_pivote: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze per-sequence label agreement across multiple data sources.

    This function assumes `df_pivote` is a "wide" table where:
    - Each row corresponds to a peptide sequence.
    - The 'sequence' column is the identifier.
    - All other columns correspond to labels assigned by different sources.

    Label encoding
    --------------
    1   → positive
    0   → negative
    2   → unlabelled (the source provides the sequence but no class label)
    999 → unknown / missing / not-applicable placeholder

    Outputs
    -------
    The function appends the following columns to the input DataFrame:

    Counts:
    - counts_1: number of positive votes
    - counts_0: number of negative votes
    - counts_unlabel: number of unlabelled annotations (2)
    - counts_unknown: number of unknown placeholders (999)

    Consensus flags:
    - positive: at least one positive and no negatives (allows unlabelled)
    - negative: at least one negative and no positives (allows unlabelled)
    - exclusive_1: only positive labels (no negatives and no unlabelled)
    - exclusive_0: only negative labels (no positives and no unlabelled)
    - only_unlabel: only unlabelled evidence (no positives and no negatives)

    Vote percentages (computed using only labeled votes 0/1):
    - percentage_0: % negative votes among labeled votes
    - percentage_1: % positive votes among labeled votes

    Parameters
    ----------
    df_pivote : pandas.DataFrame
        Pivoted DataFrame with 'sequence' and one label column per source.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with added count, flag, and percentage columns.

    Notes
    -----
    - Percentages ignore unlabelled (2) and unknown (999) entries.
    - If a sequence has zero labeled votes (no 0/1), percentages are set to 0.
    """
    # Containers for raw label counts
    counts_0 = []
    counts_1 = []
    counts_unlabel = []
    counts_unknown = []

    # Classification flags
    positive_list = []
    negative_list = []
    exclusive_1_list = []
    exclusive_0_list = []
    only_unlabel_list = []

    # Percentage of labeled votes
    percentage_0 = []
    percentage_1 = []

    # All columns except the identifier are treated as label sources
    label_columns = [c for c in df_pivote.columns if c != "sequence"]

    for idx in df_pivote.index:
        row = df_pivote.loc[idx, label_columns]

        vote_1 = (row == 1).sum()
        vote_0 = (row == 0).sum()
        vote_2 = (row == 2).sum()
        vote_999 = (row == 999).sum()

        has_1 = vote_1 > 0
        has_0 = vote_0 > 0
        has_2 = vote_2 > 0

        # High-level classification (allowing unlabelled sources)
        positive = has_1 and not has_0          # 1 or 1+2
        negative = has_0 and not has_1          # 0 or 0+2

        # Exclusive evidence (no unlabel)
        exclusive_1 = has_1 and not has_0 and not has_2
        exclusive_0 = has_0 and not has_1 and not has_2

        # Only unlabelled evidence (unknown values may still exist)
        only_unlabel = has_2 and not has_1 and not has_0

        # Store counts and flags
        counts_1.append(vote_1)
        counts_0.append(vote_0)
        counts_unlabel.append(vote_2)
        counts_unknown.append(vote_999)

        positive_list.append(positive)
        negative_list.append(negative)
        exclusive_1_list.append(exclusive_1)
        exclusive_0_list.append(exclusive_0)
        only_unlabel_list.append(only_unlabel)

        # Percentages computed only over labeled votes (0/1)
        total_labeled = vote_0 + vote_1
        percentage_0.append((vote_0 / total_labeled) * 100 if total_labeled > 0 else 0)
        percentage_1.append((vote_1 / total_labeled) * 100 if total_labeled > 0 else 0)

    # Attach results
    df_pivote["counts_1"] = counts_1
    df_pivote["counts_0"] = counts_0
    df_pivote["counts_unlabel"] = counts_unlabel
    df_pivote["counts_unknown"] = counts_unknown

    df_pivote["positive"] = positive_list
    df_pivote["negative"] = negative_list
    df_pivote["exclusive_1"] = exclusive_1_list
    df_pivote["exclusive_0"] = exclusive_0_list
    df_pivote["only_unlabel"] = only_unlabel_list

    df_pivote["percentage_0"] = percentage_0
    df_pivote["percentage_1"] = percentage_1

    return df_pivote


def categorize_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin sequences into discrete categories based on `percentage_1`.

    This function maps the per-sequence percentage of positive votes
    (`percentage_1`) into coarse bins to simplify downstream reporting and
    stratified analyses.

    Bins
    ----
    >90, 80-90, 70-80, 60-70, 50-60, 40-50, 30-40, 20-30, 10-20, >0

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a numeric column 'percentage_1'.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with an additional column 'Category_pbb'
        containing the assigned bin for each row.

    Notes
    -----
    - The final category is labeled as '>0' and includes values <= 10.
      If you want a '0-10' label (more explicit), rename that bin accordingly.
    """
    category_pbb = []

    for index in df.index:
        percentage = df.loc[index, "percentage_1"]

        if percentage > 90:
            category_pbb.append(">90")
        elif 80 < percentage <= 90:
            category_pbb.append("80-90")
        elif 70 < percentage <= 80:
            category_pbb.append("70-80")
        elif 60 < percentage <= 70:
            category_pbb.append("60-70")
        elif 50 < percentage <= 60:
            category_pbb.append("50-60")
        elif 40 < percentage <= 50:
            category_pbb.append("40-50")
        elif 30 < percentage <= 40:
            category_pbb.append("30-40")
        elif 20 < percentage <= 30:
            category_pbb.append("20-30")
        elif 10 < percentage <= 20:
            category_pbb.append("10-20")
        else:
            category_pbb.append(">0")

    df["Category_pbb"] = category_pbb
    return df
