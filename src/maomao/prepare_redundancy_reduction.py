import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D



def read_threshold_from_json(
    percentil_dir: Path,
    json_name: str = "reduction_embcos.json",
) -> float:
    """
    Read the cosine threshold from a reduction JSON file inside a percentil folder.

    Expected file
    -------------
    percentil_dir / json_name
        e.g. .../cytolysis/p40/reduction_embcos.json

    Returns
    -------
    float
        Threshold found in:
        data["strategy_params"]["threshold"]

    Notes
    -----
    - Returns np.nan if the file does not exist or the field is missing.
    """
    json_path = percentil_dir / json_name

    if not json_path.exists():
        return np.nan

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("strategy_params", {}).get("threshold", np.nan))
    except Exception:
        return np.nan


def extract_percentil_number(percentil_name: str) -> float:
    """
    Extract the numeric percentil value from folder names such as:
    - p30 -> 30.0
    - p40 -> 40.0
    - p99_5 -> 99.5
    - p99_9 -> 99.9
    - 30 -> 30.0
    - 0.5 -> 0.5

    Returns
    -------
    float
        Numeric percentil value, or np.nan if it cannot be parsed.
    """
    text = str(percentil_name).strip().lower()

    if text.startswith("p"):
        text = text[1:]

    text = text.replace("_", ".")

    try:
        return float(text)
    except ValueError:
        return np.nan


def filter_sequence(
    df_all_unique: pd.DataFrame,
    df_labels: pd.DataFrame,
    sequence_col: str = "sequence",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Filter `df_all_unique` to only sequences present in `df_labels`, and merge labels.
    """
    df_unique_labels = (
        df_labels[[sequence_col, label_col]]
        .drop_duplicates(subset=[sequence_col])
    )

    df_filtered = (
        df_all_unique[df_all_unique[sequence_col].isin(df_unique_labels[sequence_col])]
        .merge(df_unique_labels, on=sequence_col, how="left")
        .reset_index(drop=True)
    )

    return df_filtered


def build_id_sequence_label_embedding_id(
    df: pd.DataFrame,
    prefix_id: str = "pep",
    prefix_emb: str = "emb",
    sequence_col: str = "sequence",
    label_col: str = "label",
    start_index: int = 1,
    pad: int = 4,
) -> pd.DataFrame:
    """
    Create standardized IDs for sequences and their embeddings.
    """
    df_out = df[[sequence_col, label_col]].copy().reset_index(drop=True)

    idx = range(start_index, start_index + len(df_out))

    df_out.insert(0, "id", [f"{prefix_id}_{i:0{pad}d}" for i in idx])
    df_out["embedding_id"] = [f"{prefix_emb}_{i:0{pad}d}" for i in idx]

    return df_out


def general_summary(base_dir: Path | str) -> pd.DataFrame:
    """
    Summarize redundancy reduction results across (model, effect, percentil) folders.
    """
    base_dir = Path(base_dir)
    rows: list[dict] = []

    for reduced_path in base_dir.rglob("data_nr_embcos.csv"):
        percentil_dir = reduced_path.parent
        effect_dir = percentil_dir.parent
        model_dir = effect_dir.parent

        percentil = percentil_dir.name
        percentil_num = extract_percentil_number(percentil)
        effect = effect_dir.name
        model = model_dir.name

        dataset_path = effect_dir / f"dataset_{effect}.csv"
        threshold = read_threshold_from_json(percentil_dir)

        if not dataset_path.exists():
            continue

        n_original = pd.read_csv(dataset_path, usecols=["label"]).shape[0]
        n_reduced = pd.read_csv(reduced_path, usecols=["label"]).shape[0]

        removed = n_original - n_reduced
        kept_pct = (n_reduced / n_original * 100) if n_original else np.nan
        removed_pct = (removed / n_original * 100) if n_original else np.nan
        kept_fraction = (n_reduced / n_original) if n_original else np.nan

        rows.append({
            "model": model,
            "effect": effect,
            "percentil": percentil,
            "percentil_num": percentil_num,
            "threshold": threshold,
            "n_original": n_original,
            "n_reduced": n_reduced,
            "removed": removed,
            "kept_pct": kept_pct,
            "removed_pct": removed_pct,
            "kept_fraction": kept_fraction,
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "model", "effect", "percentil", "percentil_num", "threshold",
                "n_original", "n_reduced", "removed",
                "kept_pct", "removed_pct", "kept_fraction",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["model", "effect", "percentil_num"])
        .reset_index(drop=True)
    )


def label_summary(base_dir: Path | str, expected_labels=(0, 1)) -> pd.DataFrame:
    """
    Summarize redundancy reduction results per (model, effect, percentil, class label).
    """
    base_dir = Path(base_dir)
    rows: list[dict] = []

    for reduced_path in base_dir.rglob("data_nr_embcos.csv"):
        percentil_dir = reduced_path.parent
        effect_dir = percentil_dir.parent
        model_dir = effect_dir.parent

        percentil = percentil_dir.name
        percentil_num = extract_percentil_number(percentil)
        effect = effect_dir.name
        model = model_dir.name

        dataset_path = effect_dir / f"dataset_{effect}.csv"
        threshold = read_threshold_from_json(percentil_dir)

        if not dataset_path.exists():
            continue

        df_orig = pd.read_csv(dataset_path, usecols=["label"])
        df_red = pd.read_csv(reduced_path, usecols=["label"])

        orig_counts = df_orig["label"].value_counts(dropna=False)
        red_counts = df_red["label"].value_counts(dropna=False)

        all_labels = set(orig_counts.index).union(red_counts.index).union(expected_labels)

        for lab in sorted(all_labels, key=str):
            try:
                lab_out = int(float(lab))
            except Exception:
                lab_out = str(lab)

            n_original = int(orig_counts.get(lab, 0))
            n_reduced = int(red_counts.get(lab, 0))
            removed = n_original - n_reduced

            kept_pct = (n_reduced / n_original * 100) if n_original else np.nan
            removed_pct = (removed / n_original * 100) if n_original else np.nan
            kept_fraction = (n_reduced / n_original) if n_original else np.nan

            rows.append({
                "model": model,
                "effect": effect,
                "percentil": percentil,
                "percentil_num": percentil_num,
                "threshold": threshold,
                "label": lab_out,
                "n_original": n_original,
                "n_reduced": n_reduced,
                "removed": removed,
                "kept_pct": kept_pct,
                "removed_pct": removed_pct,
                "kept_fraction": kept_fraction,
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "model", "effect", "percentil", "percentil_num", "threshold", "label",
                "n_original", "n_reduced", "removed",
                "kept_pct", "removed_pct", "kept_fraction",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["model", "effect", "percentil_num", "label"])
        .reset_index(drop=True)
    )


def graph_summary(df_label_summary, save_path=None, dpi=300):
    sns.set_theme(style="whitegrid")

    df_plot = df_label_summary[["model", "effect", "label", "n_reduced"]].copy()
    df_plot["Class"] = df_plot["label"].map({0: "Negative", 1: "Positive"})
    df_plot["Class"] = df_plot["Class"].fillna(df_plot["label"].astype(str))

    model_order = (
        df_plot.groupby("model")["n_reduced"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    g = sns.catplot(
        data=df_plot,
        x="model",
        y="n_reduced",
        hue="Class",
        col="effect",
        kind="bar",
        order=model_order,
        col_wrap=3,
        height=4,
        aspect=1.35,
        palette="deep",
        sharey=False,
        legend=False,
    )

    for ax in g.axes.flatten():
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order, rotation=45, ha="right")

    g.set_titles("{col_name}")
    g.set_axis_labels("Embedding model", "Number of surviving sequences")

    legend_elements = [
        Patch(facecolor=sns.color_palette("deep")[0], label="Negative"),
        Patch(facecolor=sns.color_palette("deep")[1], label="Positive"),
    ]
    g.fig.legend(
        handles=legend_elements,
        title="Class label",
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        bbox_to_anchor=(0.5, -0.07),
    )

    g.fig.subplots_adjust(bottom=0.28, hspace=0.5)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


def graph_keep_vs_removed_pct(df_summary, save_path=None):
    effects = ["cytolysis", "cytotoxic", "hemolytic", "neurotoxic", "toxic"]
    models = sorted(df_summary["model"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    axes = axes.ravel()

    palette = ["#7487A5", "#8CB6A0"]

    for i, effect in enumerate(effects):
        ax = axes[i]

        sub = (
            df_summary[df_summary["effect"] == effect]
            .set_index("model")
            .reindex(models)
        )

        x = np.arange(len(models))
        kept_pct = sub["kept_pct"].fillna(0).values
        removed_pct = sub["removed_pct"].fillna(0).values

        ax.bar(x, kept_pct, color=palette[0], label="Kept (%)")
        ax.bar(x, removed_pct, bottom=kept_pct, color=palette[1], label="Removed (%)")

        ax.set_title(effect.title())
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")

    axes[-1].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def graph_keep_vs_removed_pct_by_label(df_label_summary, save_path=None):
    effects = ["cytolysis", "cytotoxic", "hemolytic", "neurotoxic", "toxic"]
    models = sorted(df_label_summary["model"].unique())
    labels = sorted(df_label_summary["label"].unique())

    fig, axes = plt.subplots(
        len(labels),
        len(effects),
        figsize=(24, 7 if len(labels) == 1 else 4 * len(labels)),
        constrained_layout=True,
    )

    if len(labels) == 1:
        axes = np.array([axes])

    palette = ["#7487A5", "#8CB6A0"]

    for r, lab in enumerate(labels):
        for c, effect in enumerate(effects):
            ax = axes[r, c]

            sub = (
                df_label_summary[
                    (df_label_summary["label"] == lab) &
                    (df_label_summary["effect"] == effect)
                ]
                .set_index("model")
                .reindex(models)
            )

            kept_pct = sub["kept_pct"].fillna(0).values
            removed_pct = sub["removed_pct"].fillna(0).values
            x = np.arange(len(models))

            ax.bar(x, kept_pct, color=palette[0])
            ax.bar(x, removed_pct, bottom=kept_pct, color=palette[1])

            if r == 0:
                ax.set_title(effect.title())
            if c == 0:
                ax.set_ylabel(f"Label {lab}\nPercentage (%)")

            ax.set_ylim(0, 100)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha="right")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[0]),
        plt.Rectangle((0, 0), 1, 1, color=palette[1]),
    ]
    fig.legend(
        handles,
        ["Kept (%)", "Removed (%)"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        frameon=False,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_reduction_curves_panel(
    summary_df: pd.DataFrame,
    x_col: str = "percentil_num",
    y_col: str = "kept_fraction",
    ncols: int = 1,
    figsize_per_panel: tuple = (9, 4.8),
    marker: str = "o",
):
    """
    Plot a panel of reduction curves by percentil.
    """

    required_cols = {"model", "effect", x_col, y_col}
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = summary_df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    effects = sorted(df["effect"].unique())
    models = sorted(df["model"].unique())

    model_colors = {
        "Mistral_Prot_v1_134M": "#1f77b4",
        "ankh2_ext1": "#ff7f0e",
        "ankh3_large": "#2ca02c",
        "esm2_t12_35M_UR50D": "#d62728",
        "esm2_t30_150M_UR50D": "#9467bd",
        "esm2_t33_650M_UR50D": "#8c564b",
        "esm2_t6_8M_UR50D": "#e377c2",
        "esmc_300m": "#7f7f7f",
        "kmer_k3": "#bcbd22",
        "one-hot": "#17becf",
        "prot_bert": "#aec7e8",
        "prot_t5_xl_uniref50": "#ffbb78",
    }

    n_panels = len(effects)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        sharex=False,
        sharey=True if y_col in {"kept_pct", "kept_fraction"} else False,
    )

    axes_flat = axes.flatten()

    for ax, effect in zip(axes_flat, effects):
        df_eff = df[df["effect"] == effect].copy()

        xticks = sorted(df_eff[x_col].dropna().unique())

        # posiciones equidistantes
        xtick_pos = np.arange(len(xticks))
        xtick_map = dict(zip(xticks, xtick_pos))

        for model in models:
            df_model = df_eff[df_eff["model"] == model].copy()
            if df_model.empty:
                continue

            df_model = df_model.sort_values(x_col)
            df_model["x_plot"] = df_model[x_col].map(xtick_map)

            ax.plot(
                df_model["x_plot"],
                df_model[y_col],
                marker=marker,
                linewidth=1.8,
                markersize=5.5,
                label=model,
                color=model_colors.get(model, None),
            )

        ax.set_title(effect, fontsize=14)
        ax.set_xlabel("Percentil", fontsize=11)
        ax.set_ylabel(y_col.replace("_", " "), fontsize=11)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(
            [str(int(x)) if float(x).is_integer() else str(x) for x in xticks],
            rotation=60,
            ha="right",
        )

        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.margins(x=0.08)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(effects):]:
        ax.set_visible(False)

    legend_handles = [
        Line2D(
            [0], [0],
            color=model_colors[model],
            marker=marker,
            linewidth=1.8,
            markersize=5.5,
            label=model,
        )
        for model in models
        if model in model_colors
    ]

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            frameon=True,
            facecolor="white",
            edgecolor="#e6e6e6",
            framealpha=0.8,
            fancybox=True,
            borderpad=0.6,
            labelspacing=0.4,
            title="numerical representations",
            fontsize=10,
            title_fontsize=10,
            ncol=6,
        )

    fig.subplots_adjust(hspace=0.35, wspace=0.20, bottom=0.18)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, axes


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_reduction_bars_panel(
    summary_df: pd.DataFrame,
    x_col: str = "percentil_num",
    y_col: str = "kept_fraction",
    ncols: int = 1,
    figsize_per_panel: tuple = (14, 5.5),
    group_width: float = 0.95,
):
    """
    Plot a panel of grouped bar charts by percentil.

    Each panel corresponds to one effect.
    Within each percentil, bars are grouped by model.
    """

    required_cols = {"model", "effect", x_col, y_col}
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = summary_df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    effects = sorted(df["effect"].unique())
    models = sorted(df["model"].unique())

    model_colors = {
        "Mistral_Prot_v1_134M": "#1f77b4",
        "ankh2_ext1": "#ff7f0e",
        "ankh3_large": "#2ca02c",
        "esm2_t12_35M_UR50D": "#d62728",
        "esm2_t30_150M_UR50D": "#9467bd",
        "esm2_t33_650M_UR50D": "#8c564b",
        "esm2_t6_8M_UR50D": "#e377c2",
        "esmc_300m": "#7f7f7f",
        "kmer_k3": "#bcbd22",
        "one-hot": "#17becf",
        "prot_bert": "#aec7e8",
        "prot_t5_xl_uniref50": "#ffbb78",
    }

    n_panels = len(effects)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        sharex=False,
        sharey=True if y_col in {"kept_pct", "kept_fraction"} else False,
    )

    axes_flat = axes.flatten()

    for ax, effect in zip(axes_flat, effects):
        df_eff = df[df["effect"] == effect].copy()

        xticks = sorted(df_eff[x_col].dropna().unique())
        x_pos = np.arange(len(xticks))

        n_models = len(models)
        bar_width = group_width / max(n_models, 1)

        for i, model in enumerate(models):
            df_model = df_eff[df_eff["model"] == model].copy()
            if df_model.empty:
                continue

            df_model = (
                df_model[[x_col, y_col]]
                .drop_duplicates(subset=[x_col])
                .set_index(x_col)
                .reindex(xticks)
                .reset_index()
            )

            offset = (i - (n_models - 1) / 2) * bar_width

            ax.bar(
                x_pos + offset,
                df_model[y_col],
                width=bar_width,
                label=model,
                color=model_colors.get(model, None),
                alpha=0.95,
                edgecolor=None,
                linewidth=0,
            )

        ax.set_title(effect, fontsize=14)
        ax.set_xlabel("Percentil", fontsize=11)
        ax.set_ylabel(y_col.replace("_", " "), fontsize=11)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [str(int(x)) if float(x).is_integer() else str(x) for x in xticks],
            rotation=60,
            ha="right",
        )

        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.margins(x=0.01)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    for ax in axes_flat[len(effects):]:
        ax.set_visible(False)

    legend_handles = [
        Patch(
            facecolor=model_colors[model],
            edgecolor="none",
            label=model,
        )
        for model in models
        if model in model_colors
    ]

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            frameon=True,
            facecolor="white",
            edgecolor="#e6e6e6",
            framealpha=0.9,
            fancybox=True,
            borderpad=0.6,
            labelspacing=0.4,
            title="numerical representations",
            fontsize=10,
            title_fontsize=10,
            ncol=6,
        )

    fig.subplots_adjust(hspace=0.35, wspace=0.20, bottom=0.20)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, axes