import numpy as np
import pandas as pd


def _to_numpy_1d(x):
    """
    Convert input to a clean 1D NumPy array of floats.

    Inputs
    ------
    x : array-like (list, np.ndarray, pd.Series)
        Input vector.

    Outputs
    -------
    np.ndarray
        1D float array with NaNs removed.

    Notes
    -----
    - Uses pandas.isna to remove NaNs robustly.
    - Forces dtype=float for consistent numeric computation.
    """
    arr = np.asarray(x)
    arr = arr[~pd.isna(arr)]
    return arr.astype(float)


def _set_seed(seed=None):
    """
    Create a NumPy random generator for reproducibility.

    Inputs
    ------
    seed : int or None
        Seed value. If None, RNG is non-deterministic.

    Outputs
    -------
    numpy.random.Generator
        Random number generator instance.
    """
    return np.random.default_rng(seed)


# ============================================================
# Significance labels
# ============================================================
def significance_stars(p, thresholds=None, na_symbol=""):
    """
    Convert a p-value (or q-value) into significance stars.

    Parameters
    ----------
    p : float
        P-value or adjusted p-value.
    thresholds : list[tuple[float, str]] or None, default=None
        Ordered thresholds for significance annotation.

        Default:
            p <= 0.0001 -> "****"
            p <= 0.001  -> "***"
            p <= 0.01   -> "**"
            p <= 0.05   -> "*"
            otherwise   -> "ns"

    na_symbol : str, default=""
        Symbol returned when p is NaN.

    Returns
    -------
    str
        Significance label.
    """
    if pd.isna(p):
        return na_symbol

    if thresholds is None:
        thresholds = [
            (1e-4, "****"),
            (1e-3, "***"),
            (1e-2, "**"),
            (5e-2, "*"),
        ]

    for thr, stars in thresholds:
        if p <= thr:
            return stars

    return "ns"


# ============================================================
# Multiple-testing correction (BH / FDR)
# ============================================================
def p_adjust_bh(pvals):
    """
    Benjamini–Hochberg procedure (FDR control).

    Inputs
    ------
    pvals : array-like
        Collection of raw p-values.

    Outputs
    -------
    np.ndarray
        BH-adjusted p-values (q-values), same length as pvals.

    Notes
    -----
    - Sorts p-values, applies monotone adjustment from the largest rank.
    - Caps adjusted values at 1.0.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adj = np.empty(m, dtype=float)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = (m / rank) * ranked[i]
        running_min = min(running_min, val)
        adj[i] = min(running_min, 1.0)

    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


# ============================================================
# Non-parametric effect sizes
# ============================================================
def cliffs_delta(x, y):
    """
    Cliff's delta effect size: P(X>Y) - P(X<Y).

    Inputs
    ------
    x, y : array-like
        Two independent samples.

    Outputs
    -------
    float
        Cliff's delta in [-1, 1].

    Notes
    -----
    - delta > 0 suggests x tends to be larger than y.
    - O(n_x * n_y) due to pairwise comparisons.
    """
    x = _to_numpy_1d(x)
    y = _to_numpy_1d(y)
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return (gt - lt) / (len(x) * len(y))


def vargha_delaney_A(x, y):
    """
    Vargha–Delaney A effect size: P(X>Y) + 0.5*P(X=Y).

    Inputs
    ------
    x, y : array-like
        Two independent samples.

    Outputs
    -------
    float
        A statistic in [0, 1].

    Notes
    -----
    - A = 0.5 indicates no stochastic dominance.
    - Relation to Cliff's delta: A = (delta + 1) / 2.
    - O(n_x * n_y) due to pairwise comparisons.
    """
    x = _to_numpy_1d(x)
    y = _to_numpy_1d(y)
    gt = np.sum(x[:, None] > y[None, :])
    eq = np.sum(x[:, None] == y[None, :])
    return (gt + 0.5 * eq) / (len(x) * len(y))


def effect_size_with_ci(x, y, effect="cliffs", n_boot=5000, balanced=True, seed=None):
    """
    Effect size with a 95% CI via bootstrap.

    Inputs
    ------
    x, y : array-like
        Two independent samples.
    effect : str
        'cliffs' or one of {'vdA', 'vargha_delaney', 'A'}.
    n_boot : int
        Number of bootstrap iterations.
    balanced : bool
        If True, uses subsampling WITHOUT replacement with
        n = min(len(x), len(y)) for each group (balanced groups).
        If False, uses classic bootstrap WITH replacement at original sizes.
    seed : int or None
        Random seed.

    Outputs
    -------
    dict
        {
            "effect": float,
            "ci_low": float,
            "ci_high": float,
            "bootstrap_samples": np.ndarray
        }

    Notes
    -----
    - Balanced=True is useful when groups have different sizes and you want
      effect size estimates less driven by sample-size imbalance.
    """
    rng = _set_seed(seed)
    x = _to_numpy_1d(x)
    y = _to_numpy_1d(y)

    if effect == "cliffs":
        eff_fn = cliffs_delta
    elif effect in ("vdA", "vargha_delaney", "A"):
        eff_fn = vargha_delaney_A
    else:
        raise ValueError("effect must be 'cliffs' or 'vdA'.")

    point = eff_fn(x, y)

    boot = np.empty(n_boot, dtype=float)
    if balanced:
        n = min(len(x), len(y))
        for i in range(n_boot):
            xb = rng.choice(x, size=n, replace=False)
            yb = rng.choice(y, size=n, replace=False)
            boot[i] = eff_fn(xb, yb)
    else:
        for i in range(n_boot):
            xb = rng.choice(x, size=len(x), replace=True)
            yb = rng.choice(y, size=len(y), replace=True)
            boot[i] = eff_fn(xb, yb)

    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return {
        "effect": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_samples": boot
    }


# ============================================================
# Statistics of interest (two groups)
# ============================================================
def diff_of_medians(x, y):
    """
    Difference of medians: median(x) - median(y).
    """
    x = _to_numpy_1d(x)
    y = _to_numpy_1d(y)
    return float(np.median(x) - np.median(y))


def diff_of_means(x, y):
    """
    Difference of means: mean(x) - mean(y).
    """
    x = _to_numpy_1d(x)
    y = _to_numpy_1d(y)
    return float(np.mean(x) - np.mean(y))


# ============================================================
# Permutation test
# ============================================================
def permutation_test_two_groups(x, y, stat_fn=diff_of_medians, n_perm=20000, two_sided=True, seed=None):
    """
    Permutation test for two independent groups.

    Inputs
    ------
    x, y : array-like
        Two independent samples.
    stat_fn : callable
        Statistic to compare groups (default: diff_of_medians).
    n_perm : int
        Number of permutations.
    two_sided : bool
        If True, p-value uses |perm_stat| >= |observed|.
        If False, uses perm_stat >= observed.
    seed : int or None
        Random seed.

    Outputs
    -------
    dict
        {
            "observed": float,
            "p_value": float,
            "perm_stats": np.ndarray
        }
    """
    rng = _set_seed(seed)
    x = _to_numpy_1d(x)
    y = _to_numpy_1d(y)

    observed = stat_fn(x, y)
    combined = np.concatenate([x, y])
    n_x = len(x)

    perm_stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(combined)
        perm_stats[i] = stat_fn(perm[:n_x], perm[n_x:])

    if two_sided:
        p = float(np.mean(np.abs(perm_stats) >= np.abs(observed)))
    else:
        p = float(np.mean(perm_stats >= observed))

    return {"observed": observed, "p_value": p, "perm_stats": perm_stats}


# ============================================================
# Balanced bootstrap of a statistic (stability)
# ============================================================
def balanced_bootstrap_stat(x, y, stat_fn=diff_of_medians, n_boot=5000, seed=None):
    """
    Balanced bootstrap (subsampling without replacement) for a statistic.

    Inputs
    ------
    x, y : array-like
        Two independent samples.
    stat_fn : callable
        Statistic function (e.g., diff_of_medians).
    n_boot : int
        Number of bootstrap iterations.
    seed : int or None
        Random seed.

    Outputs
    -------
    dict
        {
            "n_balanced": int,
            "bootstrap_stats": np.ndarray,
            "ci_low": float,
            "ci_high": float
        }
    """
    rng = _set_seed(seed)
    x = _to_numpy_1d(x)
    y = _to_numpy_1d(y)
    n = min(len(x), len(y))

    boot_stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        xb = rng.choice(x, size=n, replace=False)
        yb = rng.choice(y, size=n, replace=False)
        boot_stats[i] = stat_fn(xb, yb)

    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
    return {
        "n_balanced": n,
        "bootstrap_stats": boot_stats,
        "ci_low": ci_low,
        "ci_high": ci_high
    }


# ============================================================
# Helper: infer descriptor columns
# ============================================================
def infer_descriptor_columns(df, label_col, exclude=None):
    """
    Infer numeric descriptor columns from a DataFrame.

    Inputs
    ------
    df : pandas.DataFrame
        Input DataFrame.
    label_col : str
        Name of the label/target column.
    exclude : iterable or None
        Additional columns to exclude (e.g., IDs).

    Outputs
    -------
    list[str]
        List of numeric columns excluding label_col and exclude.
    """
    exclude = set(exclude or [])
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c != label_col and c not in exclude]
    return cols


# ============================================================
# Main analysis pipeline
# ============================================================
def analyze_two_groups_many_descriptors(
    df,
    label_col,
    descriptor_cols,
    stat_fn=diff_of_medians,
    effect="cliffs",
    n_perm=20000,
    n_boot=5000,
    seed=123,
    fdr_alpha=0.05,
    min_abs_effect=0.15,
    include_boot_ci_for_stat=True
):
    """
    Final pipeline: two classes, many descriptors.

    For each descriptor:
      1) Permutation test (non-parametric) on stat_fn(x, y)
      2) Optional balanced bootstrap CI for the statistic
      3) Effect size (Cliff or vdA) + balanced bootstrap CI

    Across descriptors:
      4) Benjamini–Hochberg FDR correction on permutation p-values
      5) Selection: q_bh < fdr_alpha and non-trivial effect threshold

    Outputs
    -------
    (pandas.DataFrame, pandas.DataFrame)
        out : full results table
        selected : filtered subset passing FDR + effect threshold
    """
    rng = np.random.default_rng(seed)
    labels = pd.unique(df[label_col].dropna())
    if len(labels) != 2:
        raise ValueError(
            f"Exactly 2 classes are required in {label_col}. Found: {len(labels)} -> {labels}"
        )

    lab_a, lab_b = labels[0], labels[1]

    rows = []
    for col in descriptor_cols:
        x = df.loc[df[label_col] == lab_a, col].dropna().values
        y = df.loc[df[label_col] == lab_b, col].dropna().values

        if len(x) < 2 or len(y) < 2:
            rows.append({
                "descriptor": col,
                "label_a": lab_a,
                "label_b": lab_b,
                "n_a": len(x),
                "n_b": len(y),
                "n_balanced": np.nan,
                "stat_observed": np.nan,
                "stat_ci_low": np.nan,
                "stat_ci_high": np.nan,
                "effect": np.nan,
                "effect_ci_low": np.nan,
                "effect_ci_high": np.nan,
                "p_perm": np.nan,
            })
            continue

        # Step 1: permutation test
        s1 = int(rng.integers(0, 1_000_000_000))
        perm = permutation_test_two_groups(
            x, y, stat_fn=stat_fn, n_perm=n_perm, two_sided=True, seed=s1
        )

        # Step 2: optional CI for statistic
        stat_ci_low = stat_ci_high = np.nan
        n_balanced = np.nan
        if include_boot_ci_for_stat:
            s2 = int(rng.integers(0, 1_000_000_000))
            boot = balanced_bootstrap_stat(x, y, stat_fn=stat_fn, n_boot=n_boot, seed=s2)
            stat_ci_low, stat_ci_high = boot["ci_low"], boot["ci_high"]
            n_balanced = boot["n_balanced"]

        # Step 3: effect size + CI
        s3 = int(rng.integers(0, 1_000_000_000))
        eff = effect_size_with_ci(x, y, effect=effect, n_boot=n_boot, balanced=True, seed=s3)

        rows.append({
            "descriptor": col,
            "label_a": lab_a,
            "label_b": lab_b,
            "n_a": len(x),
            "n_b": len(y),
            "n_balanced": n_balanced,
            "stat_observed": perm["observed"],
            "stat_ci_low": stat_ci_low,
            "stat_ci_high": stat_ci_high,
            "effect": eff["effect"],
            "effect_ci_low": eff["ci_low"],
            "effect_ci_high": eff["ci_high"],
            "p_perm": perm["p_value"],
        })

    out = pd.DataFrame(rows).sort_values("p_perm", na_position="last")
    out["q_bh"] = p_adjust_bh(out["p_perm"].fillna(1.0).values)

    # Significance labels
    out["sig_p_perm"] = out["p_perm"].apply(significance_stars)
    out["sig_q_bh"] = out["q_bh"].apply(significance_stars)

    # Final selection
    if effect == "cliffs":
        selected = out[
            (out["q_bh"] < fdr_alpha) &
            (out["effect"].abs() >= min_abs_effect)
        ].copy()
    else:
        selected = out[
            (out["q_bh"] < fdr_alpha) &
            ((out["effect"] - 0.5).abs() >= (min_abs_effect / 2))
        ].copy()

    return out, selected


# ============================================================
# Directionality summary table
# ============================================================
def descriptor_direction_table(
    df,
    label_col,
    descriptors,
    stat="median",
    add_effect_from_selected=None,
):
    """
    Univariate descriptor summary table with directionality.

    For each descriptor, the table reports:
      - per-class median (or mean)
      - difference between classes (class B - class A)
      - qualitative direction of change (↑ in B, ↓ in B, =)

    Optionally, effect sizes and significance metrics can be merged
    from a precomputed results table.
    """
    labels = pd.unique(df[label_col].dropna())
    if len(labels) != 2:
        raise ValueError(
            f"Exactly 2 classes are required in {label_col}. Found: {labels}"
        )

    a, b = labels[0], labels[1]
    agg_fn = np.median if stat == "median" else np.mean

    rows = []
    for d in descriptors:
        xa = df.loc[df[label_col] == a, d].dropna().values
        xb = df.loc[df[label_col] == b, d].dropna().values

        if len(xa) == 0 or len(xb) == 0:
            continue

        va = float(agg_fn(xa))
        vb = float(agg_fn(xb))
        diff = vb - va

        rows.append({
            "descriptor": d,
            f"{stat}_{a}": va,
            f"{stat}_{b}": vb,
            f"diff_{b}_minus_{a}": diff,
            "direction": (
                "↑ in B" if diff > 0 else
                ("↓ in B" if diff < 0 else "=")
            ),
            "n_a": len(xa),
            "n_b": len(xb),
        })

    out = pd.DataFrame(rows)

    if add_effect_from_selected is not None:
        cols = ["descriptor"]
        for c in [
            "effect", "effect_ci_low", "effect_ci_high",
            "p_perm", "q_bh",
            "sig_p_perm", "sig_q_bh"
        ]:
            if c in add_effect_from_selected.columns:
                cols.append(c)

        out = out.merge(
            add_effect_from_selected[cols],
            on="descriptor",
            how="left"
        )

    return out


# ============================================================
# Directionality table directly from selected_df
# ============================================================
def descriptor_direction_table_from_selected(
    selected_df: pd.DataFrame,
    diff_col: str = "stat_observed",
    label_a_col: str = "label_a",
    label_b_col: str = "label_b",
    descriptor_col: str = "descriptor",
):
    """
    Same idea as descriptor_direction_table, but works on a precomputed
    selected descriptors table that already contains label_a/label_b
    and a per-descriptor observed difference / effect.

    Direction is inferred from the sign of `diff_col`:
      >0  -> ↑ in B
      <0  -> ↓ in B
      =0  -> =
    """
    required = {descriptor_col, label_a_col, label_b_col, diff_col}
    missing = required - set(selected_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = selected_df.copy()

    out["direction"] = np.where(
        out[diff_col] > 0, "↑ in B",
        np.where(out[diff_col] < 0, "↓ in B", "=")
    )

    a = out[label_a_col].dropna().unique()
    b = out[label_b_col].dropna().unique()
    if len(a) == 1 and len(b) == 1:
        out = out.rename(columns={
            diff_col: f"diff_{b[0]}_minus_{a[0]}"
        })
    else:
        out = out.rename(columns={diff_col: "diff_B_minus_A"})

    keep_order = [
        descriptor_col, label_a_col, label_b_col,
        "n_a", "n_b", "n_balanced",
        "stat_ci_low", "stat_ci_high",
        "effect", "effect_ci_low", "effect_ci_high",
        "p_perm", "sig_p_perm",
        "q_bh", "sig_q_bh",
        "direction",
    ]

    diff_cols = [c for c in out.columns if c.startswith("diff_")]
    if diff_cols:
        diff_name = diff_cols[0]
        keep_order.insert(6, diff_name)

    keep = [c for c in keep_order if c in out.columns]
    out = out[keep].sort_values(
        by=["q_bh", "p_perm", descriptor_col],
        ascending=[True, True, True],
        kind="mergesort"
    )

    return out


# ============================================================
# Sequence filtering helper
# ============================================================
def filter_sequences(
    df_data: pd.DataFrame,
    toxic_effect: str | None = None,
    activity: str | None = None,
    exclude_label: str = "No therapeutic evidence",
) -> pd.DataFrame:
    """
    Filter sequences based on toxicity annotation criteria.

    This function performs a two-step filtering procedure:
    1) Removes sequences belonging to a specified label.
    2) Optionally restricts the dataset to sequences associated with a
       specific toxic effect, while preserving sequences without annotated
       toxic effects (i.e., missing values).
    """
    df = df_data[df_data["label"] != exclude_label].copy()

    if toxic_effect:
        toxic_effect_norm = toxic_effect.lower()

        if toxic_effect_norm != "toxic":
            df = df[
                df["toxic_effects"].str.contains(
                    toxic_effect_norm,
                    case=False,
                    na=True
                )
            ]

        if activity is not None:
            df["label"] = df["label"].replace(
                {
                    "No toxic evidence": f"No {toxic_effect_norm} evidence",
                    "Therapeutic-toxic": f"{activity.title()}-{toxic_effect_norm}",
                }
            )

    return df