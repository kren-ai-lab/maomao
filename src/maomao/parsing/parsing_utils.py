import os
import re
import time
import json
import pandas as pd
from Bio import SeqIO, Entrez
from Bio.PDB import PDBParser, PPBuilder
from pathlib import Path
from collections import Counter
from typing import Dict, Pattern, Optional, List
from Bio.SeqUtils import seq1
from io import StringIO


def export_json(path_to_export, data_to_export):
    """
    Write a Python object to a JSON file.

    Parameters
    ----------
    path_to_export : str | os.PathLike
        Output file path. If the file exists, it will be overwritten.
    data_to_export : Any
        Any JSON-serializable object (dict, list, etc.). Non-serializable objects
        are converted to string using `default=str`.

    Notes
    -----
    - `ensure_ascii=False` preserves UTF-8 characters (accents, etc.).
    - `indent=4` makes the JSON human-readable.
    - `default=str` prevents crashes with types like datetime, Path, etc.
    """
    with open(path_to_export, 'w') as doc_export:
        json.dump(
            data_to_export,
            doc_export,
            indent=4,
            default=str,
            ensure_ascii=False
        )


def create_metatada(path_metada, name_source):
    """
    Load an Excel metadata table, filter by a single source name, and return a
    normalized metadata dictionary for that source.

    Parameters
    ----------
    path_metada : str | os.PathLike
        Path to the Excel file containing metadata (must include column 'name source').
    name_source : str
        Exact value to match in the 'name source' column.

    Returns
    -------
    dict
        A dictionary with standardized keys (snake_case-like names) for the selected
        metadata fields.

    Raises
    ------
    KeyError
        If expected columns are missing in the Excel file.
    IndexError
        If `name_source` does not match any row (the function assumes at least one match).

    Notes
    -----
    This function assumes the filter returns exactly one relevant row and uses the
    first row (`[0]`) as the metadata record.
    """
    # Read full metadata table
    df_metada = pd.read_excel(path_metada)

    # Filter by source name (exact match)
    df_metada_filter = df_metada[df_metada["name source"] == name_source]

    # Keep only the relevant columns (in a fixed order)
    df_metada_filter = df_metada_filter[
        [
            'type source', 'static-dynamic', 'license',
            'year of publication', 'last update date',
            'download date', 'file format',
            'peptide property', 'dataset information',
            'unit of measurement', 'obtaining negative dataset',
            'repository or server', 'publication',
        ]
    ]

    # Rename columns to a normalized naming scheme
    df_metada_filter.columns = [
        "type source", "static-dynamic", "license",
        "publication_year", "last_update", "download_date",
        "file_format", "peptide_property", "dataset_information",
        "unit_of_measurement", "negative_examples_process",
        "source_acces", "publication_link"
    ]

    # Preserve original index as a column (then it will be ignored when building the dict)
    df_metada_filter.reset_index(inplace=True)

    dict_metadata = {}

    # Convert first (and assumed unique) row into a dictionary (excluding the old index)
    for column in df_metada_filter.columns:
        if column != "index":
            dict_metadata.update({column: df_metada_filter[column][0]})

    return dict_metadata


def read_fasta_doc(doc_fasta, description=False):
    """
    Parse a FASTA file and return sequences as a pandas DataFrame.

    Parameters
    ----------
    doc_fasta : str | os.PathLike | file-like
        Path to the FASTA file or a handle accepted by Bio.SeqIO.parse.
    description : bool, default=False
        If True, include the FASTA record description column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - always: ['id', 'sequence']
        - optionally: ['description']
    """
    matrix_data = []

    # SeqIO.parse yields SeqRecord objects
    for record in SeqIO.parse(doc_fasta, "fasta"):
        if description:
            row = {
                "id": record.id,
                "sequence": str(record.seq),
                "description": str(record.description),
            }
        else:
            row = {
                "id": record.id,
                "sequence": str(record.seq),
            }

        matrix_data.append(row)

    df_export = pd.DataFrame(matrix_data)
    return df_export


def create_metada_with_multiple_values(df_metada_filter):
    """
    Build a metadata dictionary from a filtered metadata DataFrame, collapsing multiple
    distinct values per column into a single string.

    Parameters
    ----------
    df_metada_filter : pandas.DataFrame
        Filtered metadata DataFrame (e.g., a subset of a larger metadata table).
        Each column is summarized by its unique values.

    Returns
    -------
    dict
        Metadata dictionary:
        - If a column has a single unique value, stores that value.
        - If a column has multiple unique values, stores them joined by ';'.
    """
    dict_metadata = {}

    for column in df_metada_filter.columns:
        values = df_metada_filter[column].unique().tolist()

        # If multiple unique values exist, join them into a single string
        if len(values) > 1:
            values = [str(value) for value in values]
            values = ";".join(values)
            dict_metadata.update({column: values})
        else:
            dict_metadata.update({column: values[0]})

    return dict_metadata


def read_metadata(path_data, name_source):
    """
    Read an Excel metadata table and return the filtered metadata rows for one source.

    Parameters
    ----------
    path_data : str | os.PathLike
        Path to the Excel file containing metadata.
    name_source : str
        Exact match value in the 'name source' column.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only the selected columns for the specified source.

    Notes
    -----
    Unlike `create_metatada`, this function returns the filtered DataFrame instead of a dict.
    """
    df_metada = pd.read_excel(path_data)
    df_metada_filter = df_metada[df_metada["name source"] == name_source]

    # Select the same set of metadata columns (kept in original naming)
    df_metada_filter = df_metada_filter[
        [
            'type source', 'static-dynamic', 'license',
            'year of publication', 'last update date',
            'download date', 'file format',
            'peptide property', 'dataset information',
            'unit of measurement', 'obtaining negative dataset',
            'repository or server', 'publication',
        ]
    ]
    return df_metada_filter


def processing_duplicated(df_concat, group_seq="seq", sort_key="label"):
    """
    Identify duplicated sequences and handle them depending on label consistency.

    This function groups by a sequence column (default 'seq') and checks how many times
    each sequence appears. It splits sequences into:
    - unique sequences (appear exactly once)
    - duplicated sequences with consistent labels (all rows share the same label)
    - duplicated sequences with conflicting labels (appear with more than one label)

    Parameters
    ----------
    df_concat : pandas.DataFrame
        Input DataFrame containing at least `group_seq` and `sort_key` columns.
    group_seq : str, default="seq"
        Column name used to define duplicates (typically the peptide/protein sequence).
    sort_key : str, default="label"
        Column used to check whether duplicates have consistent labels.

    Returns
    -------
    df_remove_duplicated : pandas.DataFrame
        DataFrame with one row per duplicated sequence that has a consistent label.
        Columns: [group_seq, sort_key]
    df_errors : pandas.DataFrame
        DataFrame listing sequences that have conflicting labels across duplicates.
        Column: ['sequence']
    df_unique : pandas.DataFrame
        DataFrame containing the original rows for sequences that appear exactly once.

    Notes
    -----
    - For duplicates with consistent labels, it collapses them into a single row.
    - For duplicates with mixed labels, it flags them as errors (does not resolve them).
    """
    # Count occurrences per sequence and sort by frequency (using count of sort_key column)
    grouped_data = (
        df_concat.groupby(group_seq)
        .count()
        .sort_values(by=sort_key, ascending=False)
    )
    grouped_data[group_seq] = grouped_data.index

    # Sequences that appear exactly once
    unique_sequences = grouped_data[grouped_data[sort_key] == 1]

    # Recover the original rows for unique sequences
    df_unique = pd.DataFrame()
    df_unique[group_seq] = unique_sequences[group_seq].values
    df_unique = df_unique.merge(right=df_concat, on=group_seq)

    # Sequences that appear more than once
    duplicated_sequences = grouped_data[grouped_data[sort_key] != 1]

    matrix_data = []
    error_sequences = []

    # For each duplicated sequence, check whether the labels agree
    for sequence in duplicated_sequences[group_seq].values:
        data_filter = df_concat[df_concat[group_seq] == sequence]
        labels = data_filter[sort_key].unique().tolist()

        # If there is exactly one unique label -> keep as resolved duplicate
        if len(labels) == 1:
            row = {
                group_seq: sequence,
                sort_key: labels[0],
            }
            matrix_data.append(row)
        else:
            # Conflicting labels -> mark as error
            error_sequences.append(sequence)

    df_remove_duplicated = pd.DataFrame(matrix_data)
    df_errors = pd.DataFrame()
    df_errors["sequence"] = error_sequences

    return df_remove_duplicated, df_errors, df_unique


def extract_sequence_from_pdb(pdb_file: str) -> dict:
    """
    Extract amino-acid sequences from a PDB file using Biopython's PPBuilder.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file.

    Returns
    -------
    dict
        Mapping {chain_id: sequence_string}. Only chains with at least one peptide
        built by PPBuilder are included.

    Notes
    -----
    - PPBuilder builds polypeptides based on geometry; it may skip residues/chains if
      the structure is incomplete or non-standard.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    ppb = PPBuilder()
    sequences = {}

    for model in structure:
        for chain in model:
            peptides = ppb.build_peptides(chain)
            # Concatenate peptides for the same chain (if multiple segments are detected)
            seq = "".join([p.get_sequence() for p in peptides])
            if seq:
                sequences[chain.id] = str(seq)

    return sequences


def processing_folder_PDB(path_to_pdb, df_checking):
    """
    Compare PDB filenames in a folder against IDs present in a reference DataFrame.

    Parameters
    ----------
    path_to_pdb : str | os.PathLike
        Directory containing PDB files.
    df_checking : pandas.DataFrame
        DataFrame that must contain a column 'id' with the expected IDs.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - id_data: filename stem (without extension), with '_' replaced by '|'
        - is_in_doc: boolean, True if id_data is present in df_checking['id']

    Notes
    -----
    This function assumes file names follow a convention where:
    - ID is the stem (before '.')
    - '_' should be normalized into '|'
    """
    list_pdbs_doc = os.listdir(path_to_pdb)

    # Normalize file stems into IDs
    list_ids_doc = [value.split(".")[0].replace("_", "|") for value in list_pdbs_doc]

    df_ids = pd.DataFrame()
    df_ids["id_data"] = list_ids_doc
    df_ids["is_in_doc"] = df_ids["id_data"].isin(df_checking["id"])
    return df_ids


def checking_label_in_id(text):
    """
    Infer a binary label from an identifier string.

    Parameters
    ----------
    text : str
        Input string (typically an ID) that may contain substrings like 'POS' or 'NEG'.

    Returns
    -------
    int
        1 if 'POS' is found,
        0 if 'NEG' is found,
        -1 otherwise.

    Notes
    -----
    This is a simple substring check (case-sensitive as written).
    If you want case-insensitive behavior, you can apply `text.upper()` before checks.
    """
    if "POS" in text:
        return 1
    elif "NEG" in text:
        return 0
    else:
        return -1


def parse_adam_file(input_path: str) -> pd.DataFrame:
    """
    Parse an ADAM-formatted text file into a DataFrame.

    The file is assumed to contain records starting with a line like:
        ADAM_12345<tab>...<tab>SEQUENCE_FRAGMENT
    followed by optional continuation lines containing more sequence fragments.

    Parameters
    ----------
    input_path : str
        Path to the ADAM text file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - id: ADAM identifier (e.g., 'ADAM_123')
        - description: text between id and the last column (tab-separated)
        - sequence: concatenation of sequence fragments (whitespace removed)
        - length: sequence length

    Notes
    -----
    - Continuation lines are appended to the current sequence.
    - Lines are stripped; empty lines are ignored.
    - Encoding errors are replaced (`errors="replace"`).
    """
    input_file = Path(input_path)

    records = []
    current = None

    with input_file.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Start of a new record
            if re.match(r"^ADAM_\d+", line):
                parts = line.split("\t")
                adam_id = parts[0].strip()
                seq_fragment = parts[-1].replace(" ", "").strip()
                description = "\t".join(parts[1:-1]).strip() if len(parts) > 2 else ""

                # Save previous record before starting a new one
                if current is not None:
                    current["sequence"] = "".join(current["seq_parts"])
                    del current["seq_parts"]
                    records.append(current)

                # Initialize current record
                current = {
                    "id": adam_id,
                    "description": description,
                    "seq_parts": [seq_fragment],
                }
            else:
                # Continuation line -> append to current sequence parts
                if current:
                    cont = line.replace(" ", "")
                    current["seq_parts"].append(cont)

    # Add the last record after file ends
    if current:
        current["sequence"] = "".join(current["seq_parts"])
        del current["seq_parts"]
        records.append(current)

    df = pd.DataFrame(records, columns=["id", "description", "sequence"])
    df["length"] = df["sequence"].str.len()
    return df


def annotate_peptide_bioactivities_normalized(
    df: pd.DataFrame,
    extra_patterns: Optional[Dict[str, str]] = None,
    min_auto_count: int = 15,
    collapse_hierarchy: bool = False
) -> pd.DataFrame:
    """
    Add binary bioactivity annotation columns inferred from free-text descriptions.

    The function scans a 'description' column and creates 0/1 indicator columns for
    known activity tags (e.g., antimicrobial, antiviral, toxin, hemolytic, etc.)
    using regex patterns. It can also automatically discover frequent tokens that
    "look like" activity terms and create additional columns for them.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. Must contain a 'description' column.
    extra_patterns : dict[str, str], optional
        Extra {tag: regex_pattern} pairs to merge into the built-in seed patterns.
        Useful to extend the tag set for a project-specific ontology.
    min_auto_count : int, default=15
        Minimum frequency threshold for an automatically discovered token to become
        a new indicator column.
    collapse_hierarchy : bool, default=False
        If True, merges sub-activities (antibacterial/antiviral/...) into the
        'antimicrobial' column by taking the max across them.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with extra binary columns (0/1) per detected tag.

    Raises
    ------
    KeyError
        If 'description' column is missing.

    How it works
    ------------
    1) Normalize descriptions to lowercase.
    2) Apply a curated set of regex patterns ("seed_patterns") -> tag columns.
    3) Tokenize descriptions and count frequent "activity-like" words/phrases.
    4) For frequent tokens (>= min_auto_count), create additional indicator columns.
    5) Merge alias columns into canonical ones (e.g., 'amp' into 'antimicrobial').
    6) Optionally collapse hierarchy into antimicrobial.

    Notes
    -----
    - This is rule-based annotation and depends heavily on the quality of descriptions.
    - Regex patterns are intentionally broad; false positives are possible.
    """
    if "description" not in df.columns:
        raise KeyError("Input DataFrame must contain a 'description' column.")

    out = df.copy()
    desc = out["description"].fillna("").astype(str).str.lower()

    # Curated seed patterns for common bioactivities and related terms
    seed_patterns: Dict[str, str] = {
        "antimicrobial":   r"\bantimicrobial\b|\bamp\b(?![a-z])|\bantimicrobial peptide\b",
        "antibacterial":   r"\bantibacterial\b|\banti[-\s]?bacter",
        "antiviral":       r"\bantiviral\b|\banti[-\s]?virus",
        "antifungal":      r"\bantifungal\b|\banti[-\s]?fung",
        "antiparasitic":   r"\bantiparasitic\b|\banti[-\s]?paras",
        "antiprotozoal":   r"\bantiprotozoal\b",
        "antimycobacterial": r"\banti[-\s]?mycobact",

        "bacteriocin":     r"\bbacteriocin(s)?\b",
        "defensin":        r"\bdefensin(s)?\b",

        "toxin":           r"\btoxin(s)?\b|\bneurotoxin(s)?\b|\bcytotoxin(s)?\b|\bdermato\w*toxin\w*\b",
        "hemolytic":       r"\bhemolytic\b|\bhaemolytic\b|\bhemolysin(s)?\b|\bhaemolysin(s)?\b",
        "venom":           r"\bvenom(ous)?\b|\bvenom[-\s]?peptide\b",

        "anticancer":      r"\banti[-\s]?cancer\b|\banti[-\s]?tumor\b",
        "anti_inflammatory": r"\banti[-\s]?inflamm",
        "immunomodulatory": r"\bimmunomodulat|\bimmune modulat|\bimmunostimulat|\bimmunosuppress",

        "inhibitor":       r"\binhibitor(s)?\b|\binhibit(s|ed|ing)?\b",
        "enzyme_inhibitory": r"\bprotease inhibitor|\bribonuclease inhibitor|\bkinase inhibitor",
        "binding":         r"\breceptor[-\s]?binding\b|\bprotein[-\s]?binding\b|\bbinding\b",

        "cell_penetrating": r"\bcell[-\s]?penetrating\b|\bcpp\b(?![a-z])",
        "signal":          r"\bsignal peptide\b|\bsignalling?\b|\bsignaling\b",
        "hormonal":        r"\bhormone(s)?\b|\bhormonal\b|\bneuropeptide(s)?\b|\bpeptide hormone\b",

        "antioxidant":     r"\bantioxidant(s)?\b",
        "angiogenic":      r"\banti[-\s]?angiogen|\bpro[-\s]?angiogen|angiogenic",
        "antidiabetic":    r"\banti[-\s]?diabet|\bglucose[-\s]?regulat",
        "antihypertensive": r"\banti[-\s]?hypertens|\bace[-\s]?inhibitor",
        "antibiofilm":     r"\banti[-\s]?biofilm\b|\banti[-\s]?adhes",
        "wound_healing":   r"\bwound[-\s]?healing\b|\bskin repair\b",
        "skin_related":    r"\bskin\b|\bdermato\w+",
        "chemoattractant": r"\bchemotactic\b|\bchemoattract",
        "analgesic":       r"\banalgesic\b|\bantinocicept",
    }

    # Allow users to extend/override patterns
    if extra_patterns:
        seed_patterns.update(extra_patterns)

    # Compile regex patterns once
    compiled: Dict[str, Pattern] = {
        tag: re.compile(pat, re.IGNORECASE) for tag, pat in seed_patterns.items()
    }

    # Create 0/1 columns for each seed tag
    for tag, creg in compiled.items():
        out[tag] = desc.apply(lambda s: 1 if creg.search(s) else 0)

    # Tokenization/heuristics for auto-discovered activity-like tokens
    splitter = re.compile(r"[;,\|\(\)\[\]\{\}/]+")
    wordpat = re.compile(r"[a-z][a-z\-]+")
    suffix_like = (
        "toxin", "lysin", "lytic", "cidal", "cidin", "cins", "icin",
        "mycin", "kinin", "peptide", "lectin", "insulin", "hormone"
    )
    keyword_like = {
        "defensin", "bacteriocin", "inhibitor", "binding", "venom",
        "capsid", "ribosome-inactivating", "ribonuclease", "protease",
        "kinase", "phosphatase", "transport", "carrier", "channel",
        "hemolysin", "haemolysin", "dermatoxin", "skin",
        "antimicrobial", "antiviral", "antifungal", "antibacterial"
    }

    def looks_like_activity_token(tok: str) -> Optional[str]:
        """
        Heuristic classifier: returns a normalized token string if it resembles
        an activity/functional label; otherwise None.
        """
        t = tok.strip().lower().replace("_", "-")
        if not t or len(t) < 3:
            return None
        if t in keyword_like:
            return t
        for sfx in suffix_like:
            if t.endswith(sfx):
                return t
        if any(k in t for k in ("binding", "inhibitor", "defensin", "bacteriocin", "venom")):
            return t
        if t.startswith("dermato") and "toxin" in t:
            return "dermatoxin"
        return None

    # Count candidate tokens/ngrams across descriptions
    counts = Counter()
    for s in desc:
        chunks = splitter.split(s)
        for ch in chunks:
            words = wordpat.findall(ch)
            if not words:
                continue
            # Consider 1-gram, 2-gram, 3-gram candidates
            for n in (1, 2, 3):
                if n > len(words):
                    break
                for i in range(len(words) - n + 1):
                    cand = " ".join(words[i:i + n])
                    lab = looks_like_activity_token(cand)
                    if lab:
                        counts[lab] += 1

    # Auto-create columns for frequent candidates not already present
    auto_tags = [
        lab for lab, c in counts.items()
        if c >= min_auto_count and lab not in out.columns
    ]

    for lab in sorted(auto_tags):
        token = re.escape(lab).replace(r"\ ", r"[-\s]?")
        pat = re.compile(rf"(?<![a-z0-9]){token}(?![a-z0-9])", re.IGNORECASE)
        col = re.sub(r"[^a-z0-9]+", "_", lab).strip("_")
        if col in out.columns:
            col = f"auto_{col}"
        out[col] = desc.apply(lambda s: 1 if pat.search(s) else 0)

    # Merge aliases into canonical columns (reduce redundancy)
    alias_groups: Dict[str, List[str]] = {
        "antimicrobial": [
            "antimicrobial", "anti_microbial", "antimicrobial_peptide", "amp",
            "auto_antimicrobial", "auto_amp"
        ],
        "antibacterial": [
            "antibacterial", "anti_bacterial", "antibacterial_peptide", "anti_bacter"
        ],
        "antiviral": [
            "antiviral", "anti_viral", "antiviral_peptide", "anti_virus"
        ],
        "antifungal": [
            "antifungal", "anti_fungal", "antifungal_peptide", "anti_fung"
        ],
        "antiparasitic": [
            "antiparasitic", "anti_parasitic", "anti_paras"
        ],
        "antimycobacterial": [
            "antimycobacterial", "anti_mycobacterial"
        ],

        "defensin": ["defensin", "defensins"],
        "bacteriocin": ["bacteriocin", "bacteriocins"],
        "hemolytic": ["hemolytic", "haemolytic", "hemolysin", "haemolysin"],
        "toxin": ["toxin", "neurotoxin", "cytotoxin", "dermatoxin"],
        "binding": ["binding", "receptor_binding", "protein_binding", "ligand_binding"],
        "inhibitor": ["inhibitor", "enzyme_inhibitory", "protease_inhibitor", "kinase_inhibitor"],
        "cell_penetrating": ["cell_penetrating", "cpp"],
        "anti_inflammatory": ["anti_inflammatory", "antiinflammatory"],
        "hormonal": ["hormonal", "peptide_hormone", "neuropeptide"],
        "skin_related": ["skin_related", "skin", "dermato", "dermatoxin"],
        "antibiofilm": ["antibiofilm", "anti_adhes"]
    }

    for canonical, aliases in alias_groups.items():
        present = [c for c in aliases if c in out.columns]
        if not present:
            continue
        merged = out[present].max(axis=1)
        out[canonical] = merged.astype(int)
        for col in present:
            if col != canonical:
                out.drop(columns=col, inplace=True, errors="ignore")

    # Optionally collapse antimicrobial hierarchy
    if collapse_hierarchy:
        subs = [
            c for c in ["antibacterial", "antiviral", "antifungal", "antiparasitic", "antimycobacterial"]
            if c in out.columns
        ]
        if subs:
            base = out.get("antimicrobial", 0)
            out["antimicrobial"] = pd.DataFrame(
                {"base": base, **{s: out[s] for s in subs}}
            ).max(axis=1).astype(int)

    return out


def parsing_data_on_folders(path_data, target):
    """
    Read multiple CSV files from a nested directory structure and concatenate them.

    Expected folder structure (conceptually)
    ---------------------------------------
    path_data/
        ts_1/
            tc_1/
                file1.csv
                file2.csv
            tc_2/
                ...
        ts_2/
            ...

    Parameters
    ----------
    path_data : str | os.PathLike
        Root directory containing the nested folders.
        NOTE: The code uses string concatenation like f"{path_data}{ts}",
        so `path_data` should typically end with a '/' (or you should refactor to Path).
    target : Any
        Value stored in a new column 'target' for every loaded row.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame with extra columns:
        - ts : top-level folder name
        - tc : second-level folder name
        - target : provided value
    """
    full_df_list = []

    list_ts = os.listdir(path_data)
    for ts in list_ts:
        list_tc = os.listdir(f"{path_data}{ts}")
        for tc in list_tc:
            list_files = os.listdir(f"{path_data}{ts}/{tc}")
            for doc in list_files:
                df = pd.read_csv(f"{path_data}{ts}/{tc}/{doc}")
                df["ts"] = ts
                df["tc"] = tc
                df["target"] = target
                full_df_list.append(df)

    full_df = pd.concat(full_df_list, axis=0)
    return full_df


# NCBI Entrez requires an email for API usage (polite usage policy)
Entrez.email = "david.medina@umag.cl"


def get_protein_sequence(accession, retries=3, wait=2):
    """
    Fetch a protein sequence from NCBI using an accession ID (Entrez efetch).

    Parameters
    ----------
    accession : str
        Protein accession (or GI) to fetch from the NCBI 'protein' database.
    retries : int, default=3
        Number of attempts before giving up.
    wait : int | float, default=2
        Seconds to sleep between failed attempts.

    Returns
    -------
    str | None
        Protein sequence as a string if retrieval succeeds, otherwise None.

    Notes
    -----
    - This function swallows exceptions and retries.
    - On final failure, it prints a message and returns None.
    - Consider logging the exception if you need debugging.
    """
    for attempt in range(retries):
        try:
            handle = Entrez.efetch(
                db="protein",
                id=accession,
                rettype="fasta",
                retmode="text"
            )
            seq_record = SeqIO.read(handle, "fasta")
            handle.close()
            return str(seq_record.seq)
        except Exception:
            time.sleep(wait)

    print(f"The sequences was not obtained for accesscion {accession}")
    return None


def read_apd3_txt(path_input, name_source, filename="file.txt"):
    """
    Parse a text export from APD3-like pages/files into a DataFrame.

    The function expects alternating lines like:
    - Line i:   APxxxxx
    - Line i+1: <description tokens> <SEQUENCE>

    It also skips menu/navigation lines starting with certain prefixes.

    Parameters
    ----------
    path_input : str | os.PathLike
        Base directory containing a subfolder for the source.
    name_source : str
        Subfolder name under `path_input` containing the APD3 file.
    filename : str, default="file.txt"
        Text filename to parse inside (path_input / name_source / filename).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: ['id', 'description', 'sequence'].

    Notes
    -----
    - The function assumes that for each AP record there is a second line available.
      If the file is malformed (missing the i+1 line), it may raise IndexError.
    """
    path_file = Path(path_input) / name_source / filename
    records = []

    # Prefixes to ignore (UI text, navigation)
    skip_prefixes = (
        "Found",
        "Or you can also select",
        "Search again",
        "Home"
    )

    # Read and filter lines
    with open(path_file, "r", encoding="utf-8", errors="replace") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith(skip_prefixes)
        ]

    i = 0
    while i < len(lines):
        # Ensure current line is an AP identifier like "AP00011"
        if not re.match(r"^AP\d+", lines[i]):
            i += 1
            continue

        ap_id = lines[i]
        info_seq = lines[i + 1]
        i += 2

        # Convention: last token is the sequence, previous tokens are the description
        parts = info_seq.split()
        sequence = parts[-1]
        description = " ".join(parts[:-1])

        records.append({
            "id": ap_id,
            "description": description,
            "sequence": sequence
        })

    return pd.DataFrame(records)


def read_fasta_with_strange_character(path):
    """
    Read a FASTA file robustly when it may contain encoding issues or unusual characters.

    This is a lightweight FASTA parser that:
    - treats lines starting with '>' as headers
    - concatenates subsequent lines as the sequence until the next header
    - uses `errors="replace"` to avoid crashing on invalid UTF-8 bytes

    Parameters
    ----------
    path : str | os.PathLike
        Path to the FASTA file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['id', 'sequence'], where 'id' is the raw header
        content without the leading '>'.
    """
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        seq_id = None
        seq_chunks = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous record before starting a new one
                if seq_id is not None:
                    records.append({
                        "id": seq_id,
                        "sequence": "".join(seq_chunks)
                    })
                seq_id = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)

        # Flush last record
        if seq_id is not None:
            records.append({
                "id": seq_id,
                "sequence": "".join(seq_chunks)
            })

    return pd.DataFrame(records)


def read_metadata_multiple(path_data, name_source):
    """
    Read an Excel metadata table and filter rows where 'name source' contains a substring.

    Parameters
    ----------
    path_data : str | os.PathLike
        Path to the Excel metadata file.
    name_source : str
        Substring to search for within 'name source' (case-insensitive).

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with a predefined subset of metadata columns.

    Notes
    -----
    - Uses `.str.contains(..., case=False)` for case-insensitive matching.
    - If `name_source` contains regex special characters, `.str.contains` will treat them
      as regex. If you want literal substring matching, pass `regex=False`.
    """
    df_metadata = pd.read_excel(path_data)

    df_metadata_filter = df_metadata[
        df_metadata["name source"].str.contains(
            name_source.lower(),
            case=False,
            na=False
        )
    ]

    df_metadata_filter = df_metadata_filter[
        [
            'type source', 'static-dynamic', 'license',
            'year of publication', 'last update date',
            'download date', 'file format',
            'peptide property', 'dataset information',
            'unit of measurement', 'obtaining negative dataset',
            'repository or server', 'publication'
        ]
    ]

    return df_metadata_filter


def extract_sequence_from_pdb_notstandar(pdb_file: str) -> dict:
    """
    Extract chain sequences from a PDB file by iterating residues and converting 3-letter
    residue names to 1-letter codes.

    Unlike `extract_sequence_from_pdb`, this does not rely on PPBuilder. It simply walks
    through residues and includes only standard amino acids (residue.id[0] == " ").

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file.

    Returns
    -------
    dict
        Mapping {chain_id: sequence_string}.

    Notes
    -----
    - `seq1(residue.resname)` converts 3-letter AA codes to 1-letter codes.
    - If a chain has an empty/blank ID, it defaults to "A".
    - Non-standard residues (hetero, waters, ligands) are skipped.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    sequences = {}

    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                # Only standard amino acids are represented with residue.id[0] == " "
                if residue.id[0] == " ":
                    seq.append(seq1(residue.resname))

            if seq:
                chain_id = chain.id.strip() or "A"
                sequences[chain_id] = "".join(seq)

    return sequences


def read_fasta_ntxpred(path_fasta):
    """
    Parse a FASTA-like file where headers/descriptions may span multiple lines.

    Some tools (e.g., predictors) generate FASTA where the '>' header is broken across
    multiple lines. This function reconstructs valid FASTA by:
    - joining non-sequence lines to the current header
    - keeping only uppercase sequence lines matching r'^[A-Z]+$'

    Parameters
    ----------
    path_fasta : str | os.PathLike
        Path to the input FASTA-like file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['id', 'description', 'sequence'] parsed via Bio.SeqIO.

    Notes
    -----
    - Uses `errors="ignore"` when reading lines to avoid decoding crashes.
    - After reconstruction, parsing is performed using SeqIO.parse on an in-memory string.
    """
    with open(path_fasta, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    fixed = []
    header = ""
    seq = []

    for line in lines:
        line = line.rstrip()

        if line.startswith(">"):
            # Flush previous record (header + sequence)
            if header:
                fixed.append(header)
                fixed.extend(seq)
                seq = []
            header = line

        elif re.match(r"^[A-Z]+$", line):
            # Sequence lines: keep them as-is
            seq.append(line)

        else:
            # Non-sequence line: treat as continuation of the header/description
            header += " " + line.strip()

    # Flush last record
    if header:
        fixed.append(header)
        fixed.extend(seq)

    fasta_fixed = "\n".join(fixed) + "\n"

    records = [
        {
            "id": r.id,
            "description": r.description,
            "sequence": str(r.seq)
        }
        for r in SeqIO.parse(StringIO(fasta_fixed), "fasta")
    ]

    return pd.DataFrame(records)

def parse_activities(df):
    """
    Parse and standardize peptide activity measurements.

    This function processes the ``activity`` column of a DataFrame and
    extracts structured information from measurements reported as MHC,
    LD50, HC50, or LC50.

    The function identifies the activity endpoint, comparison operator,
    numeric value, associated error, and measurement unit. Unicode
    symbols, decimal separators, scientific notation, and equivalent
    unit representations are standardized.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing an ``activity`` column with activity
        measurements expressed as text.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with the following additional columns:

        - ``activity_type``:
          Original activity text after whitespace normalization.
        - ``activity_type_clean``:
          Standardized endpoint (MHC, LD50, HC50, or LC50).
        - ``activity_relation``:
          Comparison operator associated with the measurement
          (>, >=, <, <=, or =).
        - ``activity_value``:
          Activity value converted to a floating-point number.
        - ``activity_error``:
          Reported error extracted from ``± value`` or ``[value]`` notation.
        - ``activity_unit``:
          Standardized measurement unit.

    Notes
    -----
    If no explicit comparison operator is reported before the activity
    value, equality (``=``) is assumed.

    Missing or unparsable values are returned as ``NaN``.
    """
    df_parsed = df.copy()

    # ------------------------------------------------------------------
    # 1. Clean activity text
    # ------------------------------------------------------------------
    df_parsed["activity_type"] = (
        df_parsed["activity"]
        .astype("string")
        .str.strip()
    )

    # ------------------------------------------------------------------
    # 2. Identify activity endpoint
    # ------------------------------------------------------------------
    df_parsed["activity_type_clean"] = (
        df_parsed["activity_type"]
        .str.extract(
            r"(MHC|LD50|HC50|LC50)",
            flags=re.IGNORECASE,
            expand=False,
        )
        .str.upper()
    )

    # ------------------------------------------------------------------
    # 3. Extract comparison operator
    # ------------------------------------------------------------------
    def extract_relation(text):
        """
        Extract the comparison operator associated with an activity value.
        """
        if pd.isna(text):
            return np.nan

        text = (
            str(text)
            .replace("≥", ">=")
            .replace("≤", "<=")
            .replace("−", "-")
        )

        # Remove the endpoint so that the "50" in LD50, HC50, or LC50
        # is not interpreted as the activity value.
        text_without_endpoint = re.sub(
            r"(?:MHC|LD50|HC50|LC50)",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

        number_match = re.search(
            r"\d+(?:[.,]\d+)?",
            text_without_endpoint,
        )

        if number_match is None:
            return np.nan

        operators = list(
            re.finditer(
                r">=|<=|>|<|=",
                text_without_endpoint,
            )
        )

        previous_operators = [
            match
            for match in operators
            if match.start() < number_match.start()
        ]

        if previous_operators:
            return previous_operators[-1].group()

        return "="

    df_parsed["activity_relation"] = (
        df_parsed["activity_type"]
        .apply(extract_relation)
    )

    # ------------------------------------------------------------------
    # 4. Extract activity value
    # ------------------------------------------------------------------
    df_parsed["activity_value"] = (
        df_parsed["activity_type"]
        .str.extract(
            r"(?:MHC|LD50|HC50|LC50).*?"
            r"(\d+(?:[.,]\d+)?"
            r"(?:\s*(?:x|×|\*)\s*10\s*\^?\s*[-−+]?\s*\d+)?)",
            flags=re.IGNORECASE,
            expand=False,
        )
    )

    # ------------------------------------------------------------------
    # 5. Convert activity value to numeric
    # ------------------------------------------------------------------
    def parse_number(value):
        """
        Convert standard or scientific-notation activity values to float.
        """
        if pd.isna(value):
            return np.nan

        value = (
            str(value)
            .replace("−", "-")
            .replace("×", "x")
            .replace(",", ".")
            .replace("^", "")
            .replace(" ", "")
        )

        scientific_match = re.fullmatch(
            r"([+-]?\d+(?:\.\d+)?)(?:x|\*)10([+-]?\d+)",
            value,
        )

        if scientific_match:
            base = float(scientific_match.group(1))
            exponent = int(scientific_match.group(2))

            return base * (10 ** exponent)

        try:
            return float(value)

        except ValueError:
            return np.nan

    df_parsed["activity_value"] = (
        df_parsed["activity_value"]
        .apply(parse_number)
        .astype(float)
    )

    # ------------------------------------------------------------------
    # 6. Extract error reported as ± value
    # ------------------------------------------------------------------
    error_pm = (
        df_parsed["activity_type"]
        .str.extract(
            r"±\s*(\d+(?:[.,]\d+)?)",
            expand=False,
        )
        .astype("string")
        .str.replace(",", ".", regex=False)
    )

    error_pm = pd.to_numeric(
        error_pm,
        errors="coerce",
    )

    # ------------------------------------------------------------------
    # 7. Extract error reported as [value]
    # ------------------------------------------------------------------
    error_bracket = (
        df_parsed["activity_type"]
        .str.extract(
            r"\[(\d+(?:[.,]\d+)?)\]",
            expand=False,
        )
        .astype("string")
        .str.replace(",", ".", regex=False)
    )

    error_bracket = pd.to_numeric(
        error_bracket,
        errors="coerce",
    )

    # ------------------------------------------------------------------
    # 8. Combine error values
    # ------------------------------------------------------------------
    df_parsed["activity_error"] = (
        error_pm
        .fillna(error_bracket)
        .astype(float)
    )

    # ------------------------------------------------------------------
    # 9. Extract measurement unit
    # ------------------------------------------------------------------
    df_parsed["activity_unit"] = (
        df_parsed["activity_type"]
        .str.extract(
            r"(µM|μM|uM|"
            r"µg/mL|μg/mL|ug/mL|"
            r"µg/ml|μg/ml|ug/ml|"
            r"mg/mL|mg/ml|"
            r"mg/L|mg/l|"
            r"g/mL|g/ml|"
            r"g/L|g/l|"
            r"μmol/L|µmol/L|"
            r"mM|"
            r"g\s+ml-1|"
            r"ml-1)",
            expand=False,
        )
    )

    # ------------------------------------------------------------------
    # 10. Standardize measurement units
    # ------------------------------------------------------------------
    df_parsed["activity_unit"] = (
        df_parsed["activity_unit"]
        .astype("string")
        .str.replace(r"\s+", "", regex=True)
        .replace(
            {
                "μM": "µM",
                "uM": "µM",
                "μmol/L": "µM",
                "µmol/L": "µM",
                "μg/mL": "µg/mL",
                "μg/ml": "µg/mL",
                "µg/ml": "µg/mL",
                "ug/mL": "µg/mL",
                "ug/ml": "µg/mL",
                "mg/ml": "mg/mL",
                "mg/l": "mg/L",
                "g/ml": "g/mL",
                "g/l": "g/L",
                "gml-1": "g/mL",
            }
        )
    )

    return df_parsed


def create_label(row):
    """
    Create a standardized activity label from parsed measurement fields.

    The label is constructed using the activity relation, numeric value,
    and optional error. Exact measurements are returned without an
    explicit equality sign, whereas bounded measurements retain their
    comparison operator.

    Parameters
    ----------
    row : pandas.Series
        Row containing the columns ``activity_relation``,
        ``activity_value``, and ``activity_error``.

    Returns
    -------
    str or numpy.nan
        Formatted activity label. Returns ``NaN`` when the activity value
        is missing.

    Examples
    --------
    ``= 25`` with no error -> ``25``

    ``= 25`` with error 2 -> ``25 ± 2``

    ``> 100`` with no error -> ``> 100``

    ``>= 50`` with error 5 -> ``>= 50 ± 5``
    """
    relation = row["activity_relation"]
    value = row["activity_value"]
    error = row["activity_error"]

    if pd.isna(value):
        return np.nan

    # Equality is assumed when a parsed value has no explicit relation.
    if pd.isna(relation):
        relation = "="

    value_str = f"{value:g}"

    if relation == "=":
        if pd.notna(error):
            return f"{value_str} ± {error:g}"

        return value_str

    if pd.notna(error):
        return f"{relation} {value_str} ± {error:g}"

    return f"{relation} {value_str}"

def has_annotation(value):
    """
    Return True when a modification-related field contains a usable value.

    Empty strings and common textual missing-value representations are treated
    as absent annotations.
    """
    if pd.isna(value):
        return False

    value = str(value).strip()

    return value.lower() not in {
        "",
        "none",
        "na",
        "n/a",
        "null",
    }


def describe_modifications(row):
    """
    Build a readable modification annotation for a Hemolytik 2.0 record.

    Terminal modifications, non-linear topology, D/mixed stereochemistry,
    and non-natural residue annotations are retained when present.
    """
    modifications = []

    if has_annotation(row["nter"]) and str(row["nter"]).strip() != "Free":
        modifications.append(f"N-terminal: {str(row['nter']).strip()}")

    if has_annotation(row["cter"]) and str(row["cter"]).strip() != "Free":
        modifications.append(f"C-terminal: {str(row['cter']).strip()}")

    if has_annotation(row["lyn_cyc"]) and str(row["lyn_cyc"]).strip() != "Linear":
        modifications.append(f"Topology: {str(row['lyn_cyc']).strip()}")

    if has_annotation(row["ldmix"]) and str(row["ldmix"]).strip() != "L":
        modifications.append(f"Stereochemistry: {str(row['ldmix']).strip()}")

    if has_annotation(row["non_nat"]):
        modifications.append(f"Non-natural: {str(row['non_nat']).strip()}")

    if not modifications:
        return "Unspecified modification"

    return "; ".join(modifications)

def assign_hemolytic_label(value):
    """
    Convert the Hemolytik 2.0 non-hemolytic annotation into a binary label.

    Missing ``non_hem`` values and ``Low hemolytic`` records are treated as
    positive hemolytic evidence (label 1). Explicit ``Non-hemolytic`` records
    are treated as negative evidence (label 0).

    Unexpected non-missing annotations raise an error instead of being
    silently assigned to a class.
    """
    if pd.isna(value):
        return 1

    value = str(value).strip()

    if value == "Low hemolytic":
        return 1

    if value == "Non-hemolytic":
        return 0

    raise ValueError(f"Unexpected non_hem value: {value!r}")

def prepare_regression_data(df, activity_pattern, modified=False):
    """
    Parse quantitative hemolytic measurements from row-level source records.

    Regression records are prepared independently from classification
    deduplication so that multiple measurements reported for the same peptide
    sequence are preserved.

    Parameters
    ----------
    df : pandas.DataFrame
        Source-level peptide records containing an ``activity`` column.
    modified : bool, default=False
        Whether the input contains the ``modification`` column.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Parsed valid regression records and records that could not be fully
        parsed into endpoint, numeric value, and measurement unit.
    """
    regression_raw = df.loc[
        df["activity"].astype("string").str.contains(
            activity_pattern,
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    parsed = parse_activities(regression_raw)

    parsed["label"] = parsed.apply(
        create_label,
        axis=1,
    )

    invalid_mask = (
        parsed["activity_type_clean"].isna()
        | parsed["activity_value"].isna()
        | parsed["activity_unit"].isna()
        | parsed["label"].isna()
    )

    parsing_errors = (
        parsed.loc[invalid_mask]
        .copy()
        .reset_index(drop=True)
    )

    valid = parsed.loc[~invalid_mask].copy()

    output_columns = ["sequence"]

    if modified:
        output_columns.append("modification")

    output_columns.extend(
        [
            "source",
            "label",
            "activity_unit",
            "activity_type_clean",
        ]
    )

    valid = (
        valid[output_columns]
        .rename(columns={"activity_unit": "unit"})
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return valid, parsing_errors

def select_endpoint(df, endpoint):
    """
    Return one endpoint-specific regression dataset.
    """
    return (
        df.loc[
            df["activity_type_clean"].eq(endpoint)
        ]
        .drop(columns="activity_type_clean")
        .reset_index(drop=True)
    )