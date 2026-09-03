"""
Plotting and summary utilities for peptide toxicity dataset characterization.

This module contains only the functions needed by the notebook that generates:
- dataset composition by endpoint
- ambiguity distribution
- top-level co-occurrence heatmap
- cytotoxic branch decomposition
- cytotoxic relation hierarchy
- amino acid composition panel
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import to_rgb, to_hex
import matplotlib.patheffects as pe


# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------

__all__ = [
    "get_panel_style",
    "style_axes",
    "style_legend",
    "compute_top_level_toxicity_summary",
    "plot_dataset_composition_by_endpoint_vertical",
    "plot_all_effects_cooccurrence_heatmap",
    "compute_effect_cooccurrence"
]

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _copy_with_lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of a dataframe with lowercase column names.
    """
    out = df.copy()
    out.columns = out.columns.astype(str).str.lower()
    return out

def _validate_required_columns(df: pd.DataFrame, required_cols: list[str], df_name: str = "df") -> None:
    """
    Raise an informative error if required columns are missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")

def get_panel_style(scale: float = 1.32) -> dict:
    """
    Return shared style tokens used across all figures.
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

    scale_keys = [
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
    for key in scale_keys:
        style[key] = base[key] * scale

    return style

def style_axes(ax, style: dict, grid_axis: str | None = "y", hide_top_right: bool = True) -> None:
    """
    Apply shared axis styling.
    """
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for spine_name in ("left", "bottom"):
        if spine_name in ax.spines:
            ax.spines[spine_name].set_color(style["axis_line"])
            ax.spines[spine_name].set_linewidth(style["spine_lw"])

    ax.tick_params(
        axis="both",
        labelsize=style["tick_size"],
        colors=style["text_main"],
        pad=style["tick_pad"],
    )

    if grid_axis is not None:
        ax.grid(
            axis=grid_axis,
            linestyle="--",
            linewidth=style["grid_lw"],
            alpha=style["grid_alpha"],
            color=style["grid_color"],
        )

    ax.set_axisbelow(True)

def style_legend(legend, style: dict) -> None:
    """
    Apply shared legend styling.
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

    for text in legend.get_texts():
        text.set_color(style["text_main"])
        text.set_fontsize(style["legend_size"])



# -----------------------------------------------------------------------------
# Summary builders
# -----------------------------------------------------------------------------

def compute_top_level_toxicity_summary(
    df: pd.DataFrame,
    positive_value: int = 1,
):
    """
    Compute top-level toxicity combinations and co-occurrence matrix.

    Top-level definitions
    ---------------------
    Cytotoxic = cytotoxic OR hemolytic OR cytolysis
    Neurotoxic = neurotoxic
    Embryotoxic = embryotoxic
    Ichthyotoxic = ichthyotoxic
    """
    df_lower = _copy_with_lowercase_columns(df)

    required_cols = [
        "cytotoxic",
        "hemolytic",
        "cytolysis",
        "neurotoxic",
        "embryotoxic",
        "ichthyotoxic",
        'anti_mammalian_cells'
    ]
    _validate_required_columns(df_lower, required_cols)

    effect_cols = [
        "Cytotoxic",
        "Neurotoxic",
        "Embryotoxic",
        "Ichthyotoxic",
    ]

    df_top_level = pd.DataFrame(index=df_lower.index)
    df_top_level["Cytotoxic"] = (
        df_lower["cytotoxic"].eq(positive_value)
        | df_lower["hemolytic"].eq(positive_value)
        | df_lower["cytolysis"].eq(positive_value)
    ).astype(int)
    df_top_level["Neurotoxic"] = df_lower["neurotoxic"].eq(positive_value).astype(int)
    df_top_level["Embryotoxic"] = df_lower["embryotoxic"].eq(positive_value).astype(int)
    df_top_level["Ichthyotoxic"] = df_lower["ichthyotoxic"].eq(positive_value).astype(int)

    df_top_level = df_top_level.loc[df_top_level.sum(axis=1) > 0].copy()

    combination_counts = (
        df_top_level.value_counts()
        .reset_index(name="sequence_count")
    )
    combination_counts["combination_degree"] = combination_counts[effect_cols].sum(axis=1)
    combination_counts = combination_counts.sort_values(
        by=["combination_degree", "sequence_count"],
        ascending=[True, False],
    ).reset_index(drop=True)

    X = combination_counts[effect_cols].to_numpy(dtype=int)
    weights = combination_counts["sequence_count"].to_numpy(dtype=int)
    cooccurrence_matrix = X.T @ (X * weights[:, None])
    cooccurrence_matrix = pd.DataFrame(
        cooccurrence_matrix,
        index=effect_cols,
        columns=effect_cols,
    )

    return combination_counts, cooccurrence_matrix, df_top_level


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _copy_with_lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of a dataframe with lowercase column names.
    """
    out = df.copy()
    out.columns = out.columns.astype(str).str.lower()
    return out


def _normalize_names(values) -> list[str]:
    """
    Normalize names to lowercase strings.
    """
    return [str(value).lower() for value in values]


def _title_names(values) -> list[str]:
    """
    Convert names to title case for display.
    """
    return [str(value).title() for value in values]


def _validate_required_columns(df: pd.DataFrame, required_cols: list[str], df_name: str = "df") -> None:
    """
    Raise an informative error if required columns are missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def _darken_color(color: str, amount: float = 0.78) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(np.clip(rgb * amount, 0, 1))


def _lighten_color(color: str, amount: float = 0.45) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(np.clip(rgb + (1 - rgb) * amount, 0, 1))


def _is_ambiguous_value(value) -> bool:
    """
    Identify ambiguity labels used in the processed ambiguous dataset.
    """
    if pd.isna(value) or value == 999:
        return False

    value = str(value).strip()
    return "-" in value or value in {">0", ">90", "0-10", "90-100"}


def _normalize_ambiguity_bin(value) -> str | None:
    """
    Normalize ambiguity bins before plotting.
    """
    if pd.isna(value) or value == 999:
        return None

    value = str(value).strip()

    if value in {"0-10", "0", ">0"}:
        return ">0"

    if value in {"90-100", "100", ">90"}:
        return ">90"

    return value




def plot_dataset_composition_by_endpoint_vertical(
    df_main: pd.DataFrame,
    effect_cols: list[str] | None = None,
    order: list[str] | None = None,
    endpoint_colors: dict[str, str] | None = None,
    endpoint_label_map: dict[str, str] | None = None,
    excluded_endpoints: list[str] | set[str] | tuple[str] | None = None,
    figsize: tuple[float, float] = (9, 7),
    title: str | None = None,
    ylabel: str = "Percentage of sequences",
    style: dict | None = None,
    title_fontsize: int = 14,
    tick_fontsize: int = 10,
    label_fontsize: int = 11,
    annot_fontsize: int = 10,
    legend_fontsize: int = 10,
    legend_title_fontsize: int = 10,
    legend_marker_size: int = 16,
    bold_fonts: bool = False,
    legend_loc: str = "upper center",
    legend_bbox_to_anchor: tuple[float, float] = (0.5, -0.14),
    legend_frameon: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
    small_label_threshold: float = 5.0,
    hide_label_below_pct: float | None = None,
    outside_label_color: str = "#5F6872",
    outside_label_offset: float = 2.2,
    bar_width: float = 0.72,
    column_spacing: float = 1.15,
):
    """
    Vertical stacked bar chart of dataset composition by endpoint.

    Included statuses:
    - 0 = consistent negatives
    - 1 = consistent positives
    - 2 = ambiguous
    - 3 = unlabeled

    Code 999 (no information) is excluded from the denominator.
    """
    style = get_panel_style() if style is None else style
    fontweight = "bold" if bold_fonts else "normal"

    df_main = _copy_with_lowercase_columns(df_main)

    if effect_cols is None:
        effect_cols = [
            "toxic",
            "cytotoxic",
            "hemolytic",
            "cytolysis",
            "neurotoxic",
            "embryotoxic",
            "ichthyotoxic",
            'anti_mammalian_cells'
        ]
    else:
        effect_cols = _normalize_names(effect_cols)

    if order is None:
        order = [
            "toxic",
            'anti_mammalian_cells',
            "cytotoxic",
            "embryotoxic",
            "ichthyotoxic",
            "neurotoxic",
            "hemolytic",
            "cytolysis",
        ]
    else:
        order = _normalize_names(order)

    excluded_endpoints = set() if excluded_endpoints is None else set(
        _normalize_names(excluded_endpoints)
    )

    order = [endpoint for endpoint in order if endpoint not in excluded_endpoints]
    effect_cols = [endpoint for endpoint in effect_cols if endpoint not in excluded_endpoints]

    if endpoint_colors is None:
        endpoint_colors = {
            "toxic": "#C5A3B0",
            'anti_mammalian_cells': "#CEC4DE",
            "cytotoxic": "#CEC4DE",
            "neurotoxic": "#CEC4DE",
            "embryotoxic": "#CEC4DE",
            "ichthyotoxic": "#CEC4DE",
            "hemolytic": "#DEB089",
            "cytolysis": "#DEB089",
        }
    else:
        endpoint_colors = {str(k).lower(): v for k, v in endpoint_colors.items()}

    if endpoint_label_map is None:
        endpoint_label_map = {
            "toxic": "Toxic",
            'anti_mammalian_cells': "Anti-Mammalian Cells",
            "cytotoxic": "Cytotoxic",
            "embryotoxic": "Embryotoxic",
            "ichthyotoxic": "Ichthyotoxic",
            "neurotoxic": "Neurotoxic",
            "hemolytic": "Hemolytic",
            "cytolysis": "Cytolysis",
            "cytolytic": "Cytolytic",
        }
    else:
        endpoint_label_map = {str(k).lower(): v for k, v in endpoint_label_map.items()}

    status_label_map = {
        "consistent negatives": "Consistent negatives",
        "consistent positives": "Consistent positives",
        "ambiguous": "Ambiguous",
        "unlabeled": "Unlabeled",
    }

    _validate_required_columns(df_main, effect_cols, "df_main")

    positives = (df_main[effect_cols] == 1).sum()
    negatives = (df_main[effect_cols] == 0).sum()
    ambiguous = (df_main[effect_cols] == 2).sum()
    unlabeled = (df_main[effect_cols] == 3).sum()

    summary = pd.DataFrame(
        {
            "consistent negatives": negatives,
            "consistent positives": positives,
            "ambiguous": ambiguous,
            "unlabeled": unlabeled,
        }
    ).fillna(0)

    summary = summary.reindex(order).fillna(0)

    row_totals = summary.sum(axis=1)

    if (row_totals == 0).any():
        zero_rows = [
            endpoint_label_map.get(endpoint, endpoint)
            for endpoint in row_totals[row_totals == 0].index
        ]
        raise ValueError(
            "The following endpoints have total count equal to zero "
            f"and cannot be converted to percentages: {zero_rows}"
        )

    summary_pct = summary.div(row_totals, axis=0) * 100.0

    plot_order = [
        "consistent negatives",
        "consistent positives",
        "ambiguous",
        "unlabeled",
    ]

    symbol_map = {
        "consistent negatives": "−",
        "consistent positives": "+",
        "ambiguous": "?",
        "unlabeled": "u",
    }

    hatch_map = {
        "consistent negatives": None,
        "consistent positives": None,
        "ambiguous": None,
        "unlabeled": "///",   # ayuda a distinguir unlabeled
    }

    def _format_percent(value: float) -> str:
        if abs(value - round(value)) < 0.05:
            return f"{value:.0f}%"
        return f"{value:.1f}%"

    fig, ax = plt.subplots(figsize=figsize)

    x_positions = [i * column_spacing for i in range(len(summary_pct.index))]

    for col_idx, endpoint in enumerate(summary_pct.index):
        x = x_positions[col_idx]

        base = endpoint_colors.get(endpoint, "#BFBFBF")

        color_map = {
            "consistent negatives": _darken_color(base, amount=0.72),
            "consistent positives": _darken_color(base, amount=0.90),
            "ambiguous": _lighten_color(base, amount=0.08),
            "unlabeled": _lighten_color(base, amount=0.28),
        }

        bottom_val = 0.0

        for label_type in plot_order:
            value = float(summary_pct.loc[endpoint, label_type])

            if value <= 0:
                continue

            ax.bar(
                x,
                value,
                bottom=bottom_val,
                color=color_map[label_type],
                edgecolor="white",
                linewidth=0.9,
                width=bar_width,
                hatch=hatch_map[label_type],
            )

            if hide_label_below_pct is not None and value < hide_label_below_pct:
                bottom_val += value
                continue

            label_text = f"{symbol_map[label_type]} {_format_percent(value)}"

            if value < small_label_threshold:
                y_anchor = bottom_val + value
                y_text = y_anchor + outside_label_offset

                ax.annotate(
                    label_text,
                    xy=(x, y_anchor),
                    xytext=(x, y_text),
                    ha="center",
                    va="bottom",
                    fontsize=max(7, annot_fontsize - 1),
                    color=outside_label_color,
                    alpha=0.98,
                    fontweight="bold",
                    arrowprops=dict(
                        arrowstyle="-",
                        color=outside_label_color,
                        lw=0.7,
                        alpha=0.75,
                        shrinkA=0,
                        shrinkB=2,
                    ),
                    annotation_clip=False,
                    clip_on=False,
                )
            else:
                ax.text(
                    x,
                    bottom_val + value / 2.0,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=annot_fontsize,
                    color="white",
                    alpha=0.94,
                    fontweight="bold",
                )

            bottom_val += value

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            endpoint_label_map.get(endpoint, str(endpoint).capitalize())
            for endpoint in summary_pct.index
        ],
        rotation=25,
        ha="right",
        fontsize=tick_fontsize,
        color=style["text_main"],
        fontweight=fontweight,
    )

    ax.set_xlim(min(x_positions) - 0.6, max(x_positions) + 0.6)
    ax.set_ylim(0, 106)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}%"))

    ax.set_ylabel(
        ylabel,
        fontsize=label_fontsize,
        color=style["text_main"],
        fontweight=fontweight,
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=title_fontsize,
            color=style["text_main"],
            pad=12,
            loc="left",
            fontweight=fontweight,
        )

    style_axes(ax, style, grid_axis="y")

    ax.tick_params(
        axis="x",
        labelsize=tick_fontsize,
        colors=style["text_main"],
    )

    ax.tick_params(
        axis="y",
        labelsize=tick_fontsize,
        colors=style["text_main"],
    )

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(fontweight)

    legend_handles = [
        Line2D(
            [0], [0],
            linestyle="None",
            marker=r"$+$",
            markersize=legend_marker_size,
            color="#6F7C89",
            label=status_label_map["consistent positives"],
        ),
        Line2D(
            [0], [0],
            linestyle="None",
            marker=r"$-$",
            markersize=legend_marker_size,
            color="#6F7C89",
            label=status_label_map["consistent negatives"],
        ),
        Line2D(
            [0], [0],
            linestyle="None",
            marker=r"$?$",
            markersize=legend_marker_size,
            color="#6F7C89",
            label=status_label_map["ambiguous"],
        ),
        Line2D(
            [0], [0],
            linestyle="None",
            marker=r"$u$",
            markersize=legend_marker_size,
            color="#6F7C89",
            label=status_label_map["unlabeled"],
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
        ncol=4,
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

    summary_out = summary.copy()
    summary_pct_out = summary_pct.copy()

    summary_out.index = [
        endpoint_label_map.get(endpoint, str(endpoint).capitalize())
        for endpoint in summary_out.index
    ]
    summary_pct_out.index = [
        endpoint_label_map.get(endpoint, str(endpoint).capitalize())
        for endpoint in summary_pct_out.index
    ]

    summary_out.columns = [
        status_label_map.get(col, col)
        for col in summary_out.columns
    ]
    summary_pct_out.columns = [
        status_label_map.get(col, col)
        for col in summary_pct_out.columns
    ]

    return fig, ax, summary_out, summary_pct_out






def plot_all_effects_cooccurrence_heatmap(
    cooccurrence_matrix: pd.DataFrame,
    row_totals: pd.Series | dict,
    style: dict | None = None,
    figsize: tuple[float, float] = (7.4, 6.2),
    title: str | None = None,
    cbar: bool = True,
    cbar_label: str | None = "% of row endpoint sequences also assigned to the column endpoint",
    label_map: dict[str, str] | None = None,
    effect_order: list[str] | None = None,
    tick_fontsize: int = 22,
    annot_fontsize: int = 22,
    title_fontsize: int = 22,
    cbar_tick_fontsize: int = 22,
    cbar_label_fontsize: int = 22,
    tick_rotation: int = 32,
    annot_decimals: int = 1,
    title_pad: int = 15,
    vmin: float = 0.0,
    vmax: float = 100.0,
    display_labels_title: bool = True,
    separate_diagonal: bool = True,
    diagonal_separator_color: str = "white",
    diagonal_separator_width: float = 22,
    diagonal_cell_color: str = "white",
    save_path: str | None = None,
    dpi: int = 300,
):
    """
    Draw a directional co-occurrence heatmap across all toxicity endpoints.

    Each cell shows the percentage of sequences from the row endpoint that
    are also assigned to the column endpoint.

    The diagonal can be visually separated with a white band to emphasize
    off-diagonal endpoint overlap.
    """
    style = get_panel_style() if style is None else style

    cooccurrence_matrix = cooccurrence_matrix.copy()
    cooccurrence_matrix.index = _normalize_names(cooccurrence_matrix.index)
    cooccurrence_matrix.columns = _normalize_names(cooccurrence_matrix.columns)

    if isinstance(row_totals, dict):
        row_totals = pd.Series(row_totals)

    row_totals = row_totals.copy()
    row_totals.index = _normalize_names(row_totals.index)

    if effect_order is None:
        effect_order = [
            'anti_mammalian_cells',
            "cytotoxic",
            "embryotoxic",
            "ichthyotoxic",
            "neurotoxic",
            "hemolytic",
            "cytolysis",
        ]
    else:
        effect_order = _normalize_names(effect_order)

    if label_map is None:
        label_map = {
            'anti_mammalian_cells': "Anti-Mammalian",
            "cytotoxic": "Cytotoxic",
            "neurotoxic": "Neurotoxic",
            "hemolytic": "Hemolytic",
            "cytolysis": "Cytolysis",
            "cytolytic": "Cytolytic",
            "embryotoxic": "Embryotoxic",
            "ichthyotoxic": "Ichthyotoxic",
        }
    else:
        label_map = {str(k).lower(): v for k, v in label_map.items()}

    if cooccurrence_matrix.shape[0] != cooccurrence_matrix.shape[1]:
        raise ValueError("cooccurrence_matrix must be a square matrix.")

    # Reindex to force all requested effects to appear in the heatmap.
    # Missing co-occurrence pairs are filled with 0.
    cooccurrence_matrix = cooccurrence_matrix.reindex(
        index=effect_order,
        columns=effect_order,
        fill_value=0,
    )

    labels = effect_order
    n_labels = len(labels)

    row_totals = row_totals.reindex(labels)

    if row_totals.isna().any():
        missing = [
            label_map.get(str(x), str(x).capitalize())
            for x in row_totals[row_totals.isna()].index
        ]
        raise ValueError(
            f"Missing row totals for the following endpoints: {missing}"
        )

    if (row_totals <= 0).any():
        invalid = [
            label_map.get(str(x), str(x).capitalize())
            for x in row_totals[row_totals <= 0].index
        ]
        raise ValueError(
            f"Row totals must be greater than 0 for the following endpoints: {invalid}"
        )

    overlap_pct = cooccurrence_matrix.div(row_totals, axis=0) * 100.0

    # Use copy=True to avoid read-only array issues.
    heatmap_array = overlap_pct.astype(float).to_numpy(copy=True)

    # Leave the diagonal as NaN so Matplotlib displays it in white.
    if separate_diagonal:
        np.fill_diagonal(heatmap_array, np.nan)

    if display_labels_title:
        display_labels = [
            label_map.get(lbl, str(lbl).capitalize())
            for lbl in labels
        ]
    else:
        display_labels = labels

    fig, ax = plt.subplots(figsize=figsize)

    cmap = LinearSegmentedColormap.from_list(
        "tox_all_effects_purple",
        ["#F7EEF1", "#E8CFD8", "#D9B0C0", "#CDA0B0", "#B98296"]
    )

    cmap = cmap.copy()
    cmap.set_bad(diagonal_cell_color)

    im = ax.imshow(
        heatmap_array,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_box_aspect(1)

    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))

    ax.set_xticklabels(
        display_labels,
        rotation=tick_rotation,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_fontsize,
        color=style["text_main"],
    )

    ax.set_yticklabels(
        display_labels,
        fontsize=tick_fontsize,
        color=style["text_main"],
    )

    threshold = vmax * 0.55

    for i in range(n_labels):
        for j in range(n_labels):
            if i == j and separate_diagonal:
                continue

            value = overlap_pct.iloc[i, j]
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
                zorder=3,
            )

    # White grid between cells.
    ax.set_xticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_labels, 1), minor=True)

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.35,
    )

    ax.tick_params(which="minor", bottom=False, left=False)

    # White diagonal band to visually separate both halves.
    if separate_diagonal:
        ax.plot(
            [-0.5, n_labels - 0.5],
            [-0.5, n_labels - 0.5],
            color=diagonal_separator_color,
            linewidth=diagonal_separator_width,
            solid_capstyle="butt",
            zorder=4,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        length=0,
        colors=style["text_main"],
        pad=style["tick_pad"],
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=title_fontsize,
            color=style["text_main"],
            pad=title_pad,
            loc="left",
        )

    cbar_obj = None

    if cbar:
        cbar_obj = fig.colorbar(
            im,
            ax=ax,
            fraction=0.045,
            pad=0.04,
        )

        cbar_obj.outline.set_visible(False)

        cbar_obj.ax.tick_params(
            labelsize=cbar_tick_fontsize,
            colors=style["text_main"],
            length=3,
        )

        if cbar_label is not None:
            cbar_obj.set_label(
                cbar_label,
                fontsize=cbar_label_fontsize,
                color=style["text_main"],
            )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    overlap_pct_out = overlap_pct.copy()

    if display_labels_title:
        overlap_pct_out.index = [
            label_map.get(lbl, str(lbl).capitalize())
            for lbl in overlap_pct_out.index
        ]

        overlap_pct_out.columns = [
            label_map.get(lbl, str(lbl).capitalize())
            for lbl in overlap_pct_out.columns
        ]

    return fig, ax, im, cbar_obj, overlap_pct_out

def compute_effect_cooccurrence(
    df: pd.DataFrame,
    effect_order: list[str] | None = None,
    positive_value: int | float | str = 1,
    alias_map: dict[str, list[str]] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Compute positive toxicity-endpoint co-occurrence and multi-endpoint
    statistics from the final MAOMAO sequence pivot.

    Only values equal to `positive_value` are treated as positive.
    Negative, ambiguous, unlabeled, and no-information states are treated
    as non-positive for this specific analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Sequence-level pivot containing one column per toxicity endpoint.

    effect_order : list[str] | None
        Ordered endpoint columns to include.

        By default, the broad parent endpoint `toxic` is excluded so that
        co-occurrence is calculated among the more specific toxicity
        endpoints.

    positive_value : int | float | str
        Value representing a positive endpoint annotation. Default is 1.

    alias_map : dict[str, list[str]] | None
        Mapping used only when a requested canonical endpoint column is
        missing but an alias is present.

        Example
        -------
        {"cytolysis": ["cytolytic"]}

    verbose : bool
        Print summary information when True.

    Returns
    -------
    dict
        Dictionary containing:

        - effect_binary
        - row_totals
        - positive_counts_by_endpoint
        - n_positive_endpoints_per_sequence
        - has_positive_endpoint_mask
        - annotated_mask
        - multi_endpoint_mask
        - n_positive_peptides
        - n_annotated_peptides
        - n_multi_endpoint_peptides
        - pct_multi_endpoint_peptides
        - cooccurrence_matrix
        - summary

    Notes
    -----
    `annotated_mask` and `n_annotated_peptides` are retained as legacy
    aliases for compatibility. They actually indicate sequences with at
    least one positive endpoint, not every sequence having any annotation.

    Because the final MAOMAO pivot contains hierarchy-propagated positive
    states, parent-child co-occurrence may partly reflect ontology rules.
    """

    # ---------------------------------------------------------
    # Endpoints
    # ---------------------------------------------------------

    if effect_order is None:
        effect_order = [
            "anti_mammalian_cells",
            "cytotoxic",
            "embryotoxic",
            "ichthyotoxic",
            "neurotoxic",
            "hemolytic",
            "cytolysis",
        ]

    effect_order = [
        str(effect).strip().lower()
        for effect in effect_order
    ]

    if len(effect_order) != len(set(effect_order)):
        raise ValueError(
            "effect_order contains duplicated endpoint names."
        )

    # ---------------------------------------------------------
    # Aliases
    # ---------------------------------------------------------

    if alias_map is None:
        alias_map = {
            "cytolysis": ["cytolytic"],
        }

    alias_map = {
        str(target).strip().lower(): [
            str(alias).strip().lower()
            for alias in aliases
        ]
        for target, aliases in alias_map.items()
    }

    # ---------------------------------------------------------
    # Prepare dataframe
    # ---------------------------------------------------------

    df_tmp = df.copy()

    df_tmp.columns = [
        str(column).strip().lower()
        for column in df_tmp.columns
    ]

    # Create a canonical endpoint column from an alias only if needed.
    for target_col, aliases in alias_map.items():
        if target_col in df_tmp.columns:
            continue

        for alias in aliases:
            if alias in df_tmp.columns:
                df_tmp[target_col] = df_tmp[alias]
                break

    missing_cols = [
        column
        for column in effect_order
        if column not in df_tmp.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing effect columns in dataframe: "
            f"{missing_cols}"
        )

    # Convert endpoint columns to numeric when positive_value is numeric.
    effect_data = df_tmp[effect_order].copy()

    if isinstance(positive_value, (int, float)):
        effect_data = effect_data.apply(
            pd.to_numeric,
            errors="coerce",
        )

    # ---------------------------------------------------------
    # Positive binary matrix
    # ---------------------------------------------------------

    # Rows: peptide sequences
    # Columns: toxicity endpoints
    effect_binary = effect_data.eq(positive_value).astype("int64")

    # Positive count for each endpoint.
    positive_counts_by_endpoint = effect_binary.sum(axis=0).astype(int)

    # Legacy output name retained for compatibility with the plotting
    # function already being used.
    row_totals = positive_counts_by_endpoint.copy()

    # Number of positive endpoints associated with each sequence.
    n_positive_endpoints_per_sequence = (
        effect_binary.sum(axis=1).astype(int)
    )

    # At least one positive endpoint.
    has_positive_endpoint_mask = (
        n_positive_endpoints_per_sequence.ge(1)
    )

    # More than one positive endpoint.
    multi_endpoint_mask = (
        n_positive_endpoints_per_sequence.ge(2)
    )

    n_positive_peptides = int(
        has_positive_endpoint_mask.sum()
    )

    n_multi_endpoint_peptides = int(
        multi_endpoint_mask.sum()
    )

    pct_multi_endpoint_peptides = (
        100.0
        * n_multi_endpoint_peptides
        / n_positive_peptides
        if n_positive_peptides > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Co-occurrence
    # ---------------------------------------------------------

    # Cell [i, j] is the number of sequences positive for both
    # endpoint i and endpoint j.
    #
    # int64 is essential here because int8 overflows for counts > 127.
    effect_binary_for_counts = effect_binary.astype("int64")

    cooccurrence_matrix = (
        effect_binary_for_counts.T
        .dot(effect_binary_for_counts)
        .astype("int64")
    )

    cooccurrence_matrix.index.name = "endpoint"
    cooccurrence_matrix.columns.name = "cooccurring_endpoint"

    # The diagonal must equal the number of positive sequences
    # for each endpoint.
    observed_diagonal = pd.Series(
        cooccurrence_matrix.to_numpy().diagonal(),
        index=cooccurrence_matrix.index,
        dtype="int64",
    )

    expected_diagonal = (
        positive_counts_by_endpoint
        .reindex(cooccurrence_matrix.index)
        .astype("int64")
    )

    if not observed_diagonal.equals(expected_diagonal):
        validation_table = pd.DataFrame(
            {
                "expected_positive_count": expected_diagonal,
                "observed_diagonal": observed_diagonal,
            }
        )

        validation_table["difference"] = (
            validation_table["observed_diagonal"]
            - validation_table["expected_positive_count"]
        )

        raise RuntimeError(
            "The co-occurrence diagonal does not match the positive "
            "endpoint counts.\n\n"
            f"{validation_table}"
        )

    # ---------------------------------------------------------
    # Dataset-level summary
    # ---------------------------------------------------------

    if "sequence" in df_tmp.columns:
        n_unique_sequences = int(
            df_tmp["sequence"].nunique(dropna=True)
        )
    else:
        n_unique_sequences = int(df_tmp.shape[0])

    summary = {
        "n_rows": int(df_tmp.shape[0]),
        "n_unique_sequences": n_unique_sequences,
        "n_selected_endpoints": len(effect_order),
        "selected_endpoints": effect_order.copy(),
        "n_positive_peptides": n_positive_peptides,
        "n_annotated_peptides": n_positive_peptides,
        "n_multi_endpoint_peptides": n_multi_endpoint_peptides,
        "pct_multi_endpoint_peptides": pct_multi_endpoint_peptides,
        "positive_value": positive_value,
    }

    if verbose:
        print(
            f"Sequences with at least one positive endpoint: "
            f"{n_positive_peptides:,}"
        )
        print(
            f"Sequences with multiple positive endpoints: "
            f"{n_multi_endpoint_peptides:,}"
        )
        print(
            f"Percentage among positive sequences: "
            f"{pct_multi_endpoint_peptides:.2f}%"
        )

    return {
        "effect_binary": effect_binary,

        # Both names contain the same endpoint-level counts.
        "row_totals": row_totals,
        "positive_counts_by_endpoint": positive_counts_by_endpoint,

        "n_positive_endpoints_per_sequence": (
            n_positive_endpoints_per_sequence
        ),

        "has_positive_endpoint_mask": has_positive_endpoint_mask,

        # Legacy alias retained for compatibility.
        "annotated_mask": has_positive_endpoint_mask,

        "multi_endpoint_mask": multi_endpoint_mask,

        "n_positive_peptides": n_positive_peptides,

        # Legacy alias retained for compatibility.
        "n_annotated_peptides": n_positive_peptides,

        "n_multi_endpoint_peptides": n_multi_endpoint_peptides,
        "pct_multi_endpoint_peptides": pct_multi_endpoint_peptides,
        "cooccurrence_matrix": cooccurrence_matrix,
        "summary": summary,
    }