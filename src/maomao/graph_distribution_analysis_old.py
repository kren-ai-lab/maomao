import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def get_panel_style(scale: float = 1.32):
    """
    Global style tokens for the full figure.

    Parameters
    ----------
    scale : float
        Global multiplier for typography and line widths. Increase this if the
        panel will be embedded in a manuscript and still looks too small.
    """
    base = {
        "text_main": "#1F1F1F",
        "text_soft": "#5D6973",
        "axis_line": "#CBD4DB",
        "legend_edge": "#D9DDE3",
        "grid_color": "#DDE4EA",
        "grid_alpha": 0.28,
        "grid_lw": 0.8,
        "title_size": 13.2,
        "label_size": 11.6,
        "tick_size": 10.8,
        "annot_size": 10.8,
        "cbar_tick_size": 10.2,
        "legend_size": 10.6,
        "legend_title_size": 11.2,
        "title_pad": 15,
        "spine_lw": 1.0,
        "tick_pad": 3.5,
    }

    keys_to_scale = [
        "title_size",
        "label_size",
        "tick_size",
        "annot_size",
        "cbar_tick_size",
        "legend_size",
        "legend_title_size",
        "title_pad",
        "grid_lw",
        "spine_lw",
        "tick_pad",
    ]

    style = base.copy()
    for k in keys_to_scale:
        style[k] = base[k] * scale

    return style

def style_axes(ax, style, grid_axis="y", hide_top_right=True):
    """
    Apply a unified axis style across plots.
    """
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if "left" in ax.spines:
        ax.spines["left"].set_color(style["axis_line"])
        ax.spines["left"].set_linewidth(style["spine_lw"])

    if "bottom" in ax.spines:
        ax.spines["bottom"].set_color(style["axis_line"])
        ax.spines["bottom"].set_linewidth(style["spine_lw"])

    ax.tick_params(
        axis="both",
        labelsize=style["tick_size"],
        colors=style["text_main"],
        pad=style["tick_pad"]
    )

    if grid_axis is not None:
        ax.grid(
            axis=grid_axis,
            linestyle="--",
            linewidth=style["grid_lw"],
            alpha=style["grid_alpha"],
            color=style["grid_color"]
        )

    ax.set_axisbelow(True)


def style_legend(legend, style):
    """
    Apply a unified legend style.
    """
    if legend is None:
        return

    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(style["legend_edge"])
    frame.set_linewidth(0.95)
    frame.set_alpha(0.98)

    if legend.get_title() is not None:
        legend.get_title().set_color(style["text_main"])
        legend.get_title().set_fontsize(style["legend_title_size"])

    for txt in legend.get_texts():
        txt.set_color(style["text_main"])
        txt.set_fontsize(style["legend_size"])


def compute_top_level_toxicity_summary(
    df,
    positive_value=1,
):
    """
    Compute top-level toxicity combinations and co-occurrence matrix.

    Input columns are expected in lowercase.

    Top-level definitions:
    - Cytotoxic: cytotoxic OR hemolytic OR cytolysis
    - Neurotoxic: neurotoxic
    - Embryotoxic: embryotoxic
    - Ichthyotoxic: ichthyotoxic

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with toxicity columns in lowercase.
    positive_value : int, default=1
        Value indicating presence of a toxicity effect.

    Returns
    -------
    combination_counts : pandas.DataFrame
        Table of unique toxicity combinations with counts and degree.
        Effect columns are returned in title case.
    cooccurrence_matrix : pandas.DataFrame
        Weighted co-occurrence matrix between top-level categories.
        Index and columns are returned in title case.
    df_top_level : pandas.DataFrame
        Binary matrix at top-level toxicity categories.
        Columns are returned in title case.
    """

    # Normalize column names to lowercase just in case
    df_lower = df.copy()
    df_lower.columns = df_lower.columns.str.lower()

    required_cols = [
        "cytotoxic",
        "hemolytic",
        "cytolysis",
        "neurotoxic",
        "embryotoxic",
        "ichthyotoxic",
    ]

    missing = [col for col in required_cols if col not in df_lower.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    # Output labels for plots
    effect_labels = {
        "cytotoxic": "Cytotoxic",
        "neurotoxic": "Neurotoxic",
        "embryotoxic": "Embryotoxic",
        "ichthyotoxic": "Ichthyotoxic",
    }

    # Build top-level matrix with title-case names for plotting
    df_top_level = pd.DataFrame(index=df_lower.index)

    df_top_level["Cytotoxic"] = (
        df_lower["cytotoxic"].eq(positive_value)
        | df_lower["hemolytic"].eq(positive_value)
        | df_lower["cytolysis"].eq(positive_value)
    ).astype(int)

    df_top_level["Neurotoxic"] = (
        df_lower["neurotoxic"].eq(positive_value).astype(int)
    )

    df_top_level["Embryotoxic"] = (
        df_lower["embryotoxic"].eq(positive_value).astype(int)
    )

    df_top_level["Ichthyotoxic"] = (
        df_lower["ichthyotoxic"].eq(positive_value).astype(int)
    )

    effect_cols = list(effect_labels.values())

    # Remove sequences with no top-level toxicity
    df_top_level = df_top_level.loc[
        df_top_level.sum(axis=1) > 0
    ].copy()

    # Compute combination counts
    combination_counts = (
        df_top_level
        .value_counts()
        .reset_index(name="sequence_count")
    )

    combination_counts["combination_degree"] = (
        combination_counts[effect_cols].sum(axis=1)
    )

    combination_counts = combination_counts.sort_values(
        by=["combination_degree", "sequence_count"],
        ascending=[True, False]
    ).reset_index(drop=True)

    # Compute weighted co-occurrence matrix
    X = combination_counts[effect_cols].to_numpy(dtype=int)
    weights = combination_counts["sequence_count"].to_numpy(dtype=int)

    X_weighted = X * weights[:, None]

    cooccurrence_matrix = X.T @ X_weighted

    cooccurrence_matrix = pd.DataFrame(
        cooccurrence_matrix,
        index=effect_cols,
        columns=effect_cols
    )

    return combination_counts, cooccurrence_matrix, df_top_level

def build_cytotoxic_branch_summary(df, positive_value=1):
    """
    Build summary tables for the cytotoxic branch.

    Input columns are expected in lowercase, but output labels are formatted
    for visualization.

    Returns
    -------
    internal_counts : pd.DataFrame
        Internal mutually exclusive decomposition of the cytotoxic branch.
    relation_counts : pd.DataFrame
        Co-occurrence of cytotoxic subtypes with other top-level categories.
    data : pd.DataFrame
        Filtered binary table used for the analysis.

    Definitions
    -----------
    Cytotoxic branch = cytotoxic OR hemolytic OR cytolysis

    Internal subtypes:
    - Cytotoxic only = cytotoxic == 1, hemolytic == 0, cytolysis == 0
    - Hemolytic only = hemolytic == 1, cytolysis == 0
    - Cytolysis only = cytolysis == 1, hemolytic == 0
    - Hemolytic and cytolysis = hemolytic == 1, cytolysis == 1
    """

    # ------------------------------------------------------------
    # Normalize input column names
    # ------------------------------------------------------------
    df = df.copy()
    df.columns = df.columns.str.lower()

    required_cols = [
        "cytotoxic",
        "hemolytic",
        "cytolysis",
        "neurotoxic",
        "embryotoxic",
        "ichthyotoxic",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    # ------------------------------------------------------------
    # Binarize selected columns
    # ------------------------------------------------------------
    data = pd.DataFrame(index=df.index)

    for col in required_cols:
        data[col] = df[col].eq(positive_value).astype(int)

    # ------------------------------------------------------------
    # Define cytotoxic branch
    # ------------------------------------------------------------
    data["cyto_branch"] = (
        (data["cytotoxic"] == 1)
        | (data["hemolytic"] == 1)
        | (data["cytolysis"] == 1)
    ).astype(int)

    # Keep only branch members
    data = data.loc[data["cyto_branch"] == 1].copy()

    # ------------------------------------------------------------
    # Mutually exclusive internal categories
    # ------------------------------------------------------------
    conditions = [
        (data["hemolytic"] == 1) & (data["cytolysis"] == 1),
        (data["hemolytic"] == 1) & (data["cytolysis"] == 0),
        (data["hemolytic"] == 0) & (data["cytolysis"] == 1),
        (
            (data["cytotoxic"] == 1)
            & (data["hemolytic"] == 0)
            & (data["cytolysis"] == 0)
        ),
    ]

    choices = [
        "Hemolytic and cytolysis",
        "Hemolytic only",
        "Cytolysis only",
        "Cytotoxic only",
    ]

    data["cyto_subtype"] = np.select(
        conditions,
        choices,
        default="Other cytotoxic"
    )

    subtype_order = [
        "Cytotoxic only",
        "Hemolytic only",
        "Cytolysis only",
        "Hemolytic and cytolysis",
        "Other cytotoxic",
    ]

    # ------------------------------------------------------------
    # Internal composition
    # ------------------------------------------------------------
    internal_counts = (
        data["cyto_subtype"]
        .value_counts()
        .reindex(subtype_order)
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    internal_counts.columns = ["Subtype", "Count"]

    # ------------------------------------------------------------
    # Relations with other top-level branches
    # ------------------------------------------------------------
    relation_targets = [
        "neurotoxic",
        "embryotoxic",
        "ichthyotoxic"
    ]

    relation_counts = (
        data.groupby("cyto_subtype")[relation_targets]
        .sum()
        .reindex(subtype_order)
        .fillna(0)
        .astype(int)
    )

    # Rename relation columns for plotting
    relation_counts.columns = [
        col.title() for col in relation_counts.columns
    ]

    # ------------------------------------------------------------
    # Keep only subtypes with at least one sequence
    # ------------------------------------------------------------
    valid_subtypes = internal_counts.loc[
        internal_counts["Count"] > 0,
        "Subtype"
    ].tolist()

    internal_counts = internal_counts[
        internal_counts["Subtype"].isin(valid_subtypes)
    ].copy()

    relation_counts = relation_counts.loc[valid_subtypes].copy()

    # ------------------------------------------------------------
    # Return data with title-case endpoint columns
    # ------------------------------------------------------------
    data_out = data.copy()

    rename_cols = {
        "cytotoxic": "Cytotoxic",
        "hemolytic": "Hemolytic",
        "cytolysis": "Cytolysis",
        "neurotoxic": "Neurotoxic",
        "embryotoxic": "Embryotoxic",
        "ichthyotoxic": "Ichthyotoxic",
    }

    data_out = data_out.rename(columns=rename_cols)

    return internal_counts, relation_counts, data_out

def plot_dataset_composition_by_endpoint(
    df_main,
    df_amb,
    effect_cols_main=None,
    effect_cols_amb=None,
    order=None,
    endpoint_colors=None,
    excluded_endpoints=None,
    figsize=(7.2, 5.2),
    title=None,
    ylabel="Percentage of sequences",
    style=None,

    # font sizes
    title_fontsize=14,
    tick_fontsize=10,
    label_fontsize=11,
    annot_fontsize=11,
    legend_fontsize=10,
    legend_title_fontsize=10,
    legend_marker_size=18,

    # font style
    bold_fonts=False,

    # text/layout params
    tick_rotation=28,

    # legend params
    legend_loc="upper left",
    legend_bbox_to_anchor=(1.02, 1.0),
    legend_frameon=True,

    save_path=None,
    dpi=300
):
    """
    Draw a standalone stacked bar chart showing dataset composition by endpoint.

    Input toxicity columns are expected in lowercase, but endpoint labels are
    displayed in title case in the plot.

    Percentages are computed within each endpoint.
    """

    style = get_panel_style() if style is None else style
    fontweight = "bold" if bold_fonts else "normal"

    # ------------------------------------------------------------------
    # Normalize input dataframe columns
    # ------------------------------------------------------------------
    df_main = df_main.copy()
    df_amb = df_amb.copy()

    df_main.columns = df_main.columns.str.lower()
    df_amb.columns = df_amb.columns.str.lower()

    # ------------------------------------------------------------------
    # Internal endpoint names: lowercase
    # ------------------------------------------------------------------
    if effect_cols_main is None:
        effect_cols_main = [
            "toxic",
            "hemolytic",
            "cytotoxic",
            "neurotoxic",
            "cytolysis",
            "embryotoxic",
            "ichthyotoxic"
        ]
    else:
        effect_cols_main = [str(x).lower() for x in effect_cols_main]

    if effect_cols_amb is None:
        effect_cols_amb = [
            "toxic",
            "hemolytic",
            "cytotoxic",
            "neurotoxic"
        ]
    else:
        effect_cols_amb = [str(x).lower() for x in effect_cols_amb]

    if order is None:
        order = [
            "toxic",
            "cytotoxic",
            "embryotoxic",
            "ichthyotoxic",
            "neurotoxic",
            "hemolytic",
            "cytolysis"
        ]
    else:
        order = [str(x).lower() for x in order]

    if excluded_endpoints is None:
        excluded_endpoints = set()
    else:
        excluded_endpoints = {str(x).lower() for x in excluded_endpoints}

    # ------------------------------------------------------------------
    # Colors can be given in lowercase or title case
    # ------------------------------------------------------------------
    if endpoint_colors is None:
        endpoint_colors = {
            "toxic": "#A797C7",
            "cytotoxic": "#8CB5A0",
            "neurotoxic": "#8CB5A0",
            "embryotoxic": "#8CB5A0",
            "ichthyotoxic": "#8CB5A0",
            "hemolytic": "#92A9CD",
            "cytolysis": "#92A9CD",
        }
    else:
        endpoint_colors = {
            str(k).lower(): v
            for k, v in endpoint_colors.items()
        }

    # ------------------------------------------------------------------
    # Apply exclusions
    # ------------------------------------------------------------------
    order = [ep for ep in order if ep not in excluded_endpoints]
    effect_cols_main = [
        ep for ep in effect_cols_main
        if ep not in excluded_endpoints
    ]

    # ------------------------------------------------------------------
    # Validate columns
    # ------------------------------------------------------------------
    missing_main = [
        col for col in effect_cols_main
        if col not in df_main.columns
    ]

    if missing_main:
        raise ValueError(
            f"Missing required columns in df_main: {missing_main}"
        )

    missing_amb = [
        col for col in effect_cols_amb
        if col not in df_amb.columns
    ]

    if missing_amb:
        raise ValueError(
            f"Missing required columns in df_amb: {missing_amb}"
        )

    # ------------------------------------------------------------------
    # Count consistent positives and negatives
    # ------------------------------------------------------------------
    positives = (df_main[effect_cols_main] == 1).sum()
    negatives = (df_main[effect_cols_main] == 0).sum()

    def is_ambiguous(x):
        if pd.isna(x) or x == 999:
            return False

        x = str(x).strip()

        return (
            "-" in x
            or x in [">0", ">90", "0-10", "90-100"]
        )

    ambiguous_counts = {}

    for col in effect_cols_amb:
        if col in excluded_endpoints:
            continue

        ambiguous_counts[col] = df_amb[col].apply(is_ambiguous).sum()

    ambiguous = pd.Series(ambiguous_counts)

    # ------------------------------------------------------------------
    # Build summary table using lowercase internal endpoint names
    # ------------------------------------------------------------------
    summary = pd.DataFrame({
        "consistent negatives": negatives,
        "consistent positives": positives,
        "ambiguous": ambiguous,
    }).fillna(0)

    summary = summary.reindex(order).fillna(0)

    row_totals = summary.sum(axis=1)

    if (row_totals == 0).any():
        zero_rows = row_totals[row_totals == 0].index.tolist()
        zero_rows = [x.title() for x in zero_rows]

        raise ValueError(
            "Endpoints with zero total counts cannot be converted "
            f"to percentages: {zero_rows}"
        )

    summary_pct = summary.div(row_totals, axis=0) * 100.0

    def darken_color(color, amount=0.78):
        rgb = np.array(mcolors.to_rgb(color))
        return mcolors.to_hex(np.clip(rgb * amount, 0, 1))

    def lighten_color(color, amount=0.45):
        rgb = np.array(mcolors.to_rgb(color))
        return mcolors.to_hex(np.clip(rgb + (1 - rgb) * amount, 0, 1))

    plot_order = [
        "consistent negatives",
        "consistent positives",
        "ambiguous"
    ]

    symbol_map = {
        "consistent negatives": "−",
        "consistent positives": "+",
        "ambiguous": "?",
    }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    min_icon_height = 7.0

    for i, endpoint in enumerate(summary_pct.index):
        base = endpoint_colors.get(endpoint, "#BFBFBF")

        color_map = {
            "consistent negatives": darken_color(base, amount=0.72),
            "consistent positives": base,
            "ambiguous": lighten_color(base, amount=0.3),
        }

        bottom_val = 0.0

        for label_type in plot_order:
            value = float(summary_pct.loc[endpoint, label_type])

            ax.bar(
                i,
                value,
                bottom=bottom_val,
                color=color_map[label_type],
                edgecolor="white",
                linewidth=0.85,
                width=0.78
            )

            if value >= min_icon_height:
                ax.text(
                    i,
                    bottom_val + value / 2.0,
                    symbol_map[label_type],
                    ha="center",
                    va="center",
                    fontsize=annot_fontsize,
                    color="white",
                    alpha=0.78,
                    fontweight="bold"
                )

            bottom_val += value

    # Display endpoints in title case
    ax.set_xticks(range(len(summary_pct.index)))
    ax.set_xticklabels(
        [x.title() for x in summary_pct.index],
        rotation=tick_rotation,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_fontsize,
        color=style["text_main"],
        fontweight=fontweight
    )

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))

    ax.set_ylabel(
        ylabel,
        fontsize=label_fontsize,
        color=style["text_main"],
        fontweight=fontweight
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=title_fontsize,
            color=style["text_main"],
            pad=15,
            loc="left",
            fontweight=fontweight
        )

    style_axes(ax, style, grid_axis="y")

    ax.tick_params(
        axis="y",
        labelsize=tick_fontsize,
        colors=style["text_main"]
    )

    ax.tick_params(
        axis="x",
        labelsize=tick_fontsize,
        length=0,
        colors=style["text_main"]
    )

    for label in ax.get_yticklabels():
        label.set_fontweight(fontweight)

    for label in ax.get_xticklabels():
        label.set_fontweight(fontweight)

    legend_handles = [
        Line2D(
            [0], [0],
            linestyle="None",
            marker=r"$+$",
            markersize=legend_marker_size,
            color="#6F7C89",
            label="Consistent positives"
        ),
        Line2D(
            [0], [0],
            linestyle="None",
            marker=r"$-$",
            markersize=legend_marker_size,
            color="#6F7C89",
            label="Consistent negatives"
        ),
        Line2D(
            [0], [0],
            linestyle="None",
            marker=r"$?$",
            markersize=legend_marker_size,
            color="#6F7C89",
            label="Ambiguous"
        ),
    ]

    legend = ax.legend(
        handles=legend_handles,
        title="Label status",
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        frameon=legend_frameon,
        fancybox=True,
        borderpad=0.5,
        labelspacing=0.45,
        handletextpad=0.5,
        handlelength=1.0,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=3
    )

    style_legend(legend, style)

    legend.get_title().set_fontsize(legend_title_fontsize)
    legend.get_title().set_fontweight(fontweight)

    for text in legend.get_texts():
        text.set_fontsize(legend_fontsize)
        text.set_fontweight(fontweight)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Return tables with title-case endpoint names
    # ------------------------------------------------------------------
    summary_out = summary.copy()
    summary_pct_out = summary_pct.copy()

    summary_out.index = [x.title() for x in summary_out.index]
    summary_pct_out.index = [x.title() for x in summary_pct_out.index]

    return fig, ax, summary_out, summary_pct_out

def plot_ambiguity_distribution(
    df,
    effect_cols=None,
    legend_labels=None,
    bins_order=None,
    colors=None,
    excluded_effects=None,
    figsize=(7.2, 5.2),
    title=None,
    xlabel="% of sources labeling sequence as positive",
    ylabel="Percentage of ambiguous sequences",
    style=None,

    # font sizes
    title_fontsize=14,
    tick_fontsize=10,
    label_fontsize=11,
    legend_fontsize=10,
    legend_title_fontsize=10,

    # layout/text
    tick_rotation=20,
    xlabel_pad=20,
    ylabel_pad=10,
    title_pad=14,

    # bar appearance
    total_group_width=0.92,
    bar_linewidth=0.6,
    x_margin=0.03,

    # legend
    legend_bbox_to_anchor=(0.5, -0.14),
    legend_ncol=None,
    legend_frameon=True,
    legend_handlelength=1.4,
    legend_handleheight=1.2,
    legend_handletextpad=0.6,
    legend_borderpad=0.6,
    legend_labelspacing=0.6,

    save_path=None,
    dpi=300
):
    """
    Draw ambiguity distribution as grouped bars (standalone figure).

    Bars are shown as percentages within each effect, not absolute counts.
    """

    style = get_panel_style() if style is None else style

    if effect_cols is None:
        effect_cols = ["toxic", "cytotoxic", "neurotoxic", "hemolytic"]

    if legend_labels is None:
        legend_labels = {
            "toxic": "Toxic",
            "cytotoxic": "Cytotoxic",
            "neurotoxic": "Neurotoxic",
            "hemolytic": "Hemolytic"
        }

    if bins_order is None:
        bins_order = [
            ">0", "10-20", "20-30", "30-40", "40-50",
            "50-60", "60-70", "70-80", "80-90", ">90"
        ]

    if colors is None:
        colors = {
            "toxic": "#A797C7",
            "cytotoxic": "#8CB5A0",
            "neurotoxic": "#8CB5A0",
            "hemolytic": "#92A9CD",
        }

    if excluded_effects is None:
        excluded_effects = set()
    else:
        excluded_effects = set(excluded_effects)

    effect_cols = [col for col in effect_cols if col not in excluded_effects]

    if len(effect_cols) == 0:
        raise ValueError("No effects remain after applying excluded_effects.")

    def normalize_bin(x):
        if pd.isna(x) or x == 999:
            return None
        x = str(x).strip()
        if x in ["0-10", "0", ">0"]:
            return ">0"
        if x in ["90-100", "100", ">90"]:
            return ">90"
        return x

    # counts per effect
    counts_per_effect = {}
    for col in effect_cols:
        temp = df[col].apply(normalize_bin)
        counts = temp.value_counts().reindex(bins_order).fillna(0)
        counts_per_effect[col] = counts

    plot_df = pd.DataFrame(counts_per_effect).fillna(0)

    # convert counts to percentages within each effect
    effect_totals = plot_df.sum(axis=0)
    zero_effects = effect_totals[effect_totals == 0].index.tolist()
    if zero_effects:
        raise ValueError(
            f"Effects with zero ambiguous counts cannot be converted to percentages: {zero_effects}"
        )

    plot_df_pct = plot_df.div(effect_totals, axis=1) * 100.0

    # reverse x-axis order
    bins_order_reversed = bins_order[::-1]
    plot_df = plot_df.reindex(bins_order_reversed)
    plot_df_pct = plot_df_pct.reindex(bins_order_reversed)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(plot_df_pct))
    n_effects = len(effect_cols)
    bar_width = total_group_width / n_effects

    for i, col in enumerate(effect_cols):
        offset = (i - (n_effects - 1) / 2) * bar_width

        ax.bar(
            x + offset,
            plot_df_pct[col].values,
            width=bar_width,
            label=legend_labels.get(col, col),
            color=colors.get(col, "#BFBFBF"),
            edgecolor=(1, 1, 1, 0.85),
            linewidth=bar_linewidth
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df_pct.index.astype(str),
        rotation=tick_rotation,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_fontsize,
        color=style["text_main"]
    )

    ax.set_ylim(0, 100)
    ax.margins(x=x_margin)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))

    ax.set_ylabel(
        ylabel,
        fontsize=label_fontsize,
        color=style["text_main"],
        labelpad=ylabel_pad
    )

    ax.set_xlabel(
        xlabel,
        fontsize=label_fontsize,
        color=style["text_main"],
        labelpad=xlabel_pad
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=title_fontsize,
            color=style["text_main"],
            pad=title_pad,
            loc="left"
        )

    style_axes(ax, style, grid_axis="y")

    # force tick sizes after style_axes, in case style_axes overwrites them
    ax.tick_params(axis="y", labelsize=tick_fontsize, colors=style["text_main"])
    ax.tick_params(axis="x", labelsize=tick_fontsize, length=0, colors=style["text_main"])

    for label in ax.get_xticklabels():
        label.set_fontsize(tick_fontsize)
        label.set_color(style["text_main"])

    for label in ax.get_yticklabels():
        label.set_fontsize(tick_fontsize)
        label.set_color(style["text_main"])

    if legend_ncol is None:
        legend_ncol = min(len(effect_cols), 4)

    legend = ax.legend(
        title="Effect",
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        frameon=legend_frameon,
        fancybox=True,
        borderpad=legend_borderpad,
        labelspacing=legend_labelspacing,
        handlelength=legend_handlelength,
        handleheight=legend_handleheight,
        handletextpad=legend_handletextpad,
        loc="upper center",
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol
    )

    style_legend(legend, style)

    # force legend sizes after style_legend, in case style_legend overwrites them
    legend.get_title().set_fontsize(legend_title_fontsize)
    legend.get_title().set_color(style["text_main"])

    for text in legend.get_texts():
        text.set_fontsize(legend_fontsize)
        text.set_color(style["text_main"])

    fig.subplots_adjust(bottom=0.22)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, plot_df, plot_df_pct

def plot_top_level_cooccurrence_heatmap(
    cooccurrence_matrix,
    row_totals,
    style=None,
    figsize=(6.8, 5.8),
    title=None,
    cbar=True,
    cbar_label="% of row category also in column category",

    # font sizes
    tick_fontsize=10,
    annot_fontsize=11,
    title_fontsize=14,
    cbar_tick_fontsize=10,
    cbar_label_fontsize=10,

    # text/layout
    tick_rotation=28,
    annot_decimals=1,
    title_pad=12,

    # diagonal options
    show_diagonal_text=False,
    diagonal_text_color="#6E6E6E",
    diagonal_fill_color="#EEF2EC",
    mask_diagonal=False,

    # color scale
    vmin=0.0,
    vmax=100.0,

    # label formatting
    display_labels_title=True,

    save_path=None,
    dpi=300
):
    """
    Draw a full directional overlap heatmap as a standalone figure.

    Each cell (i, j) represents:
        100 * cooccurrence_matrix[i, j] / row_totals[i]

    This answers:
        "Of the sequences in the row category, what percentage are also
        in the column category?"

    Parameters
    ----------
    cooccurrence_matrix : pd.DataFrame
        Square matrix of overlap counts between categories.

    row_totals : pd.Series or dict
        Total number of sequences in each row category. Must match the
        row labels of cooccurrence_matrix.

    display_labels_title : bool, default=True
        If True, display endpoint labels in title case in the plot.
        Internal calculations are still done using lowercase labels.
    """

    style = get_panel_style() if style is None else style

    # ------------------------------------------------------------
    # Normalize labels internally
    # ------------------------------------------------------------
    cooccurrence_matrix = cooccurrence_matrix.copy()

    cooccurrence_matrix.index = [
        str(x).lower() for x in cooccurrence_matrix.index
    ]

    cooccurrence_matrix.columns = [
        str(x).lower() for x in cooccurrence_matrix.columns
    ]

    if isinstance(row_totals, dict):
        row_totals = pd.Series(row_totals)

    row_totals = row_totals.copy()
    row_totals.index = [
        str(x).lower() for x in row_totals.index
    ]

    # ------------------------------------------------------------
    # Validate matrix
    # ------------------------------------------------------------
    if cooccurrence_matrix.shape[0] != cooccurrence_matrix.shape[1]:
        raise ValueError("cooccurrence_matrix must be square.")

    if list(cooccurrence_matrix.index) != list(cooccurrence_matrix.columns):
        raise ValueError(
            "cooccurrence_matrix must have matching row and column labels."
        )

    labels = cooccurrence_matrix.index.tolist()
    n = len(labels)

    row_totals = row_totals.reindex(labels)

    if row_totals.isna().any():
        missing = row_totals[row_totals.isna()].index.tolist()
        missing = [x.title() for x in missing]

        raise ValueError(f"Missing row totals for: {missing}")

    if (row_totals <= 0).any():
        invalid = row_totals[row_totals <= 0].index.tolist()
        invalid = [x.title() for x in invalid]

        raise ValueError(f"Row totals must be > 0 for: {invalid}")

    # ------------------------------------------------------------
    # Directional percentage matrix
    # ------------------------------------------------------------
    overlap_pct = cooccurrence_matrix.div(row_totals, axis=0) * 100.0

    heatmap_data = overlap_pct.astype(float).copy()

    if mask_diagonal:
        np.fill_diagonal(heatmap_data.values, np.nan)

    # ------------------------------------------------------------
    # Display labels
    # ------------------------------------------------------------
    if display_labels_title:
        display_labels = [x.title() for x in labels]
    else:
        display_labels = labels

    fig, ax = plt.subplots(figsize=figsize)

    cmap = LinearSegmentedColormap.from_list(
        "tox_level2",
        ["#F7FAF4", "#E6EEE0", "#B8D2C3", "#8CB5A0", "#5F8D78"]
    )

    im = ax.imshow(heatmap_data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_box_aspect(1)

    # ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    ax.set_xticklabels(
        display_labels,
        rotation=tick_rotation,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_fontsize,
        color=style["text_main"]
    )

    ax.set_yticklabels(
        display_labels,
        fontsize=tick_fontsize,
        color=style["text_main"]
    )

    # annotations
    threshold = vmax * 0.55

    for i in range(n):
        for j in range(n):
            value = overlap_pct.iloc[i, j]

            if i == j:
                if mask_diagonal:
                    continue

                if diagonal_fill_color is not None:
                    rect = plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=diagonal_fill_color,
                        edgecolor="white",
                        linewidth=1.35,
                        zorder=2
                    )
                    ax.add_patch(rect)

                if show_diagonal_text:
                    ax.text(
                        j,
                        i,
                        f"{value:.{annot_decimals}f}%",
                        ha="center",
                        va="center",
                        fontsize=annot_fontsize,
                        fontweight="bold",
                        color=diagonal_text_color,
                        zorder=3
                    )

                continue

            text_color = "white" if value > threshold else "#4F6F5F"

            ax.text(
                j,
                i,
                f"{value:.{annot_decimals}f}%",
                ha="center",
                va="center",
                fontsize=annot_fontsize,
                fontweight="bold",
                color=text_color,
                zorder=3
            )

    # grid
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    # remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        length=0,
        colors=style["text_main"],
        pad=style["tick_pad"]
    )

    # title
    if title is not None:
        ax.set_title(
            title,
            fontsize=title_fontsize,
            color=style["text_main"],
            pad=title_pad,
            loc="left"
        )

    # colorbar
    cbar_obj = None

    if cbar:
        cbar_obj = fig.colorbar(
            im,
            ax=ax,
            fraction=0.045,
            pad=0.04
        )

        cbar_obj.outline.set_visible(False)

        cbar_obj.ax.tick_params(
            labelsize=cbar_tick_fontsize,
            colors=style["text_main"],
            length=3
        )

        if cbar_label is not None:
            cbar_obj.set_label(
                cbar_label,
                fontsize=cbar_label_fontsize,
                color=style["text_main"]
            )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # Return overlap_pct with title-case labels if desired
    if display_labels_title:
        overlap_pct_out = overlap_pct.copy()
        overlap_pct_out.index = [x.title() for x in overlap_pct_out.index]
        overlap_pct_out.columns = [x.title() for x in overlap_pct_out.columns]
    else:
        overlap_pct_out = overlap_pct

    return fig, ax, im, cbar_obj, overlap_pct_out

def plot_internal_decomposition_tiles(
    internal_counts,
    style=None,
    figsize=(5.2, 5.2),
    title=None,
    cbar=True,
    cbar_label="Percentage",

    # font sizes
    title_fontsize=14,
    tile_label_fontsize=11,
    value_fontsize=12,
    cbar_tick_fontsize=10,
    cbar_label_fontsize=10,

    # text/layout
    title_pad=12,
    value_decimals=1,

    save_path=None,
    dpi=300
):
    """
    Draw a standalone 2x2 tile plot for internal subtype decomposition.

    Values are converted to percentages relative to the total Count.
    """

    style = get_panel_style() if style is None else style

    required_cols = {"Subtype", "Count"}
    if not required_cols.issubset(internal_counts.columns):
        raise ValueError("internal_counts must contain columns: 'Subtype' and 'Count'")

    plot_df = internal_counts.copy()

    desired_order = [
        "Cytotoxic only",
        "Hemolytic only",
        "Cytolysis only",
        "Hemolytic and cytolysis",
    ]

    plot_df = (
        plot_df.set_index("Subtype")
        .reindex(desired_order)
        .dropna(subset=["Count"])
        .reset_index()
    )

    if plot_df.empty:
        raise ValueError("No valid rows available to plot.")

    total_count = plot_df["Count"].sum()
    if total_count <= 0:
        raise ValueError("Total Count must be greater than zero to convert values to percentages.")

    plot_df["Percentage"] = plot_df["Count"] / total_count * 100.0

    subtypes = plot_df["Subtype"].tolist()
    percentages = plot_df["Percentage"].tolist()

    fig, ax = plt.subplots(figsize=figsize)

    cmap = LinearSegmentedColormap.from_list(
        "tox_level3_tiles",
        ["#F7FAFE", "#E1EBFF", "#BDD0EA", "#92A9CD", "#667FA8"]
    )

    vmax = max(percentages) if percentages else 1
    norm = Normalize(vmin=0, vmax=max(1, vmax))

    positions = [(0, 1), (1, 1), (0, 0), (1, 0)]

    label_map = {
        "Cytotoxic only": "Cytotoxic only",
        "Hemolytic only": "Hemolytic only",
        "Cytolysis only": "Cytolysis only",
        "Hemolytic and cytolysis": "Hemolytic +\ncytolysis",
    }

    for idx, (subtype, pct) in enumerate(zip(subtypes, percentages)):
        x, y = positions[idx]
        color = cmap(norm(pct))

        rect = Rectangle(
            (x, y),
            1,
            1,
            facecolor=color,
            edgecolor="white",
            linewidth=2.4
        )
        ax.add_patch(rect)

        text_color = "white" if pct > vmax * 0.55 else "#4F6B8A"

        ax.text(
            x + 0.5,
            y + 0.68,
            label_map.get(subtype, subtype),
            ha="center",
            va="center",
            fontsize=tile_label_fontsize,
            fontweight="semibold",
            color=text_color,
            linespacing=1.08
        )

        ax.text(
            x + 0.5,
            y + 0.30,
            f"{pct:.{value_decimals}f}%",
            ha="center",
            va="center",
            fontsize=value_fontsize,
            fontweight="bold",
            color=text_color
        )

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is not None:
        ax.set_title(
            title,
            fontsize=title_fontsize,
            color=style["text_main"],
            pad=title_pad,
            loc="left"
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_obj = None
    if cbar:
        cbar_obj = fig.colorbar(
            sm,
            ax=ax,
            fraction=0.045,
            pad=0.04
        )
        cbar_obj.outline.set_visible(False)
        cbar_obj.ax.tick_params(
            labelsize=cbar_tick_fontsize,
            colors=style["text_main"],
            length=3
        )

        if cbar_label is not None:
            cbar_obj.set_label(
                cbar_label,
                fontsize=cbar_label_fontsize,
                color=style["text_main"]
            )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, sm, cbar_obj, plot_df

def plot_cytotoxic_relation_heatmap_hierarchy(
    relation_counts,
    annotate=True,
    base_color="#5E9FA2CC",
    light_color="#FAFBFC",
    style=None,
    figsize=(5.8, 5.2),
    title=None,
    cbar=True,
    cbar_label="Percentage",

    # font sizes
    title_fontsize=14,
    tick_fontsize=10,
    annot_fontsize=11,
    cbar_tick_fontsize=10,
    cbar_label_fontsize=10,

    # text/layout
    tick_rotation=28,
    title_pad=12,
    annot_decimals=1,

    save_path=None,
    dpi=300
):
    """
    Draw a standalone cytotoxic relation heatmap.

    Values are converted to percentages by row, so each row sums to 100%.
    """

    style = get_panel_style() if style is None else style

    rel = relation_counts.copy()
    if rel.empty:
        raise ValueError("relation_counts is empty.")

    values = rel.to_numpy(dtype=float)

    row_sums = values.sum(axis=1)
    if np.any(row_sums <= 0):
        zero_rows = rel.index[row_sums <= 0].tolist()
        raise ValueError(
            f"Rows with zero total cannot be converted to percentages: {zero_rows}"
        )

    values_pct = values / row_sums[:, np.newaxis] * 100.0
    rel_pct = pd.DataFrame(values_pct, index=rel.index, columns=rel.columns)

    vmax = 100.0

    r, g, b = to_rgb(base_color)
    mid_color = (0.65 + 0.35 * r, 0.65 + 0.35 * g, 0.65 + 0.35 * b)

    cmap = LinearSegmentedColormap.from_list(
        "single_hierarchy_mix",
        [light_color, mid_color, base_color],
        N=256
    )

    norm = Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(values_pct, cmap=cmap, norm=norm, aspect="auto")
    ax.set_box_aspect(1)

    ax.set_xticks(np.arange(rel.shape[1]))
    ax.set_yticks(np.arange(rel.shape[0]))

    ax.set_xticklabels(
        rel.columns,
        fontsize=tick_fontsize,
        color=style["text_main"]
    )
    ax.set_yticklabels(
        rel.index,
        fontsize=tick_fontsize,
        color=style["text_main"]
    )
    plt.setp(ax.get_xticklabels(), rotation=tick_rotation, ha="right")

    ax.set_xticks(np.arange(-0.5, rel.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rel.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0, pad=style["tick_pad"] + 0.8)

    if annotate:
        for i in range(rel.shape[0]):
            for j in range(rel.shape[1]):
                val = values_pct[i, j]
                text_color = "white" if val > vmax * 0.58 else "#3E7274"
                ax.text(
                    j,
                    i,
                    f"{val:.{annot_decimals}f}%",
                    ha="center",
                    va="center",
                    fontsize=annot_fontsize,
                    fontweight="bold",
                    color=text_color
                )

    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is not None:
        ax.set_title(
            title,
            fontsize=title_fontsize,
            color=style["text_main"],
            pad=title_pad,
            loc="left"
        )

    cbar_obj = None
    if cbar:
        cbar_obj = fig.colorbar(
            im,
            ax=ax,
            fraction=0.045,
            pad=0.04
        )
        cbar_obj.outline.set_visible(False)
        cbar_obj.ax.tick_params(
            labelsize=cbar_tick_fontsize,
            colors=style["text_main"],
            length=3
        )

        if cbar_label is not None:
            cbar_obj.set_label(
                cbar_label,
                fontsize=cbar_label_fontsize,
                color=style["text_main"]
            )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, im, cbar_obj, rel_pct

def plot_aa_frag_barplot_ax(
    df: pd.DataFrame,
    ax,
    stats_df: pd.DataFrame | None = None,
    aa_prefix: str = "aa_frac_",
    label_col: str = "label",
    class_colors: tuple[str, str] | None = None,
    default_color: str = "#9E9E9E",
    sig_column: str = "sig_q_bh",
    star_y_offset_ratio: float = 0.035,
    star_fontsize: int | None = None,
    ylabel: str = "AA fragment frequency",
    xlabel: str = "Residue",
    show_legend: bool = True,
    style=None
) -> pd.DataFrame:


    """
    Draw amino acid composition barplot on a provided axis.
    """
    style = get_panel_style() if style is None else style
    if star_fontsize is None:
        star_fontsize = style["annot_size"] + 0.6

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' was not found in df.")

    aa_cols = [c for c in df.columns if c.startswith(aa_prefix)]
    if not aa_cols:
        raise ValueError(f"No columns found starting with prefix '{aa_prefix}'.")

    aa_cols_sorted = sorted(aa_cols)

    mean_df = (
        df.groupby(label_col)[aa_cols_sorted]
        .mean()
        .T
        .reset_index()
        .rename(columns={"index": "Residue"})
    )

    mean_df = mean_df.melt(
        id_vars="Residue",
        var_name="Label",
        value_name="AA_frag"
    )

    label_name_map = {0: "Negative", 1: "Positive"}
    mean_df["Label_name"] = mean_df["Label"].map(label_name_map)
    hue_order = ["Negative", "Positive"]

    if class_colors is not None:
        if len(class_colors) != 2:
            raise ValueError("class_colors must contain exactly 2 colors.")
        palette_final = {
            "Negative": class_colors[0],
            "Positive": class_colors[1],
        }
    else:
        palette_final = {
            "Negative": default_color,
            "Positive": default_color,
        }

    sig_map = {}
    if stats_df is not None:
        tmp = stats_df.copy()
        tmp["descriptor"] = tmp["descriptor"].astype(str)
        tmp = tmp[tmp["descriptor"].isin(aa_cols_sorted)]
        tmp[sig_column] = (
            tmp[sig_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"ns": "", "NS": "", "n.s.": "", "N.S.": ""})
        )
        sig_map = dict(zip(tmp["descriptor"], tmp[sig_column]))

    residue_order = aa_cols_sorted
    xtick_labels = [r.replace(aa_prefix, "") for r in residue_order]

    sns.barplot(
        data=mean_df,
        x="Residue",
        y="AA_frag",
        hue="Label_name",
        hue_order=hue_order,
        palette=palette_final,
        ax=ax,
        edgecolor="white",
        linewidth=0.7
    )

    ax.set_ylabel(ylabel, fontsize=style["label_size"], color=style["text_main"])
    ax.set_xlabel(xlabel, fontsize=style["label_size"], color=style["text_main"])
    ax.set_xticklabels(
        xtick_labels,
        rotation=0,
        fontsize=style["tick_size"],
        color=style["text_main"]
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}"))

    style_axes(ax, style, grid_axis="y")

    if sig_map:
        y_max = mean_df["AA_frag"].max()
        y_offset = y_max * star_y_offset_ratio if y_max > 0 else 0.01

        for i, residue in enumerate(residue_order):
            stars = sig_map.get(residue, "")
            if stars == "":
                continue

            group_max = mean_df.loc[mean_df["Residue"] == residue, "AA_frag"].max()

            ax.text(
                i,
                group_max + y_offset,
                stars,
                ha="center",
                va="bottom",
                fontsize=star_fontsize,
                fontweight="bold",
                color="#7E7E7E"
            )

        ax.set_ylim(0, y_max + y_max * 0.16 if y_max > 0 else 1)

    if show_legend:
        legend = ax.legend(
            title="Class",
            fontsize=style["legend_size"],
            frameon=True,
            fancybox=True,
            borderpad=0.5,
            labelspacing=0.42,
            handletextpad=0.5,
            loc="upper right"
        )
        style_legend(legend, style)
    else:
        existing_legend = ax.get_legend()
        if existing_legend is not None:
            existing_legend.remove()

    return mean_df

