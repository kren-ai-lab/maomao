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
import matplotlib.patheffects as pe


# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------

__all__ = [
    "get_panel_style",
    "style_axes",
    "style_legend",
    "compute_top_level_toxicity_summary",
    "plot_dataset_composition_by_endpoint_horizontal",
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



def plot_dataset_composition_by_endpoint_horizontal(
    df_main: pd.DataFrame,
    df_amb: pd.DataFrame,
    effect_cols_main: list[str] | None = None,
    effect_cols_amb: list[str] | None = None,
    order: list[str] | None = None,
    endpoint_colors: dict[str, str] | None = None,
    endpoint_label_map: dict[str, str] | None = None,
    excluded_endpoints: list[str] | set[str] | tuple[str] | None = None,
    figsize: tuple[float, float] = (7.5, 4.8),
    title: str | None = None,
    xlabel: str = "Percentage of sequences",
    style: dict | None = None,
    title_fontsize: int = 14,
    tick_fontsize: int = 10,
    label_fontsize: int = 11,
    annot_fontsize: int = 10,
    legend_fontsize: int = 10,
    legend_title_fontsize: int = 10,
    legend_marker_size: int = 18,
    bold_fonts: bool = False,
    legend_loc: str = "lower center",
    legend_bbox_to_anchor: tuple[float, float] = (0.5, -0.22),
    legend_frameon: bool = True,
    save_path: str | None = None,
    dpi: int = 300,

    # New visual controls.
    small_label_threshold: float = 5.0,
    hide_label_below_pct: float | None = None,
    outside_label_color: str = "#5F6872",
    outside_label_offset: float = 1.4,
    bar_height: float = 0.62,
    row_spacing: float = 1.12,
):
    """
    Draw a horizontal stacked bar chart showing dataset composition by toxicity endpoint.

    The visual labels are displayed in English.

    Parameters added:
    - small_label_threshold:
        Percentages below this value are placed outside the bar.
    - hide_label_below_pct:
        If set, percentages below this value are not annotated.
        Example: use 1.0 to hide labels below 1%.
    - outside_label_color:
        Color used for labels outside the bar and their guide lines.
    - outside_label_offset:
        Horizontal distance between the small segment and its external label.
    - bar_height:
        Height of each horizontal bar.
    - row_spacing:
        Vertical distance between bar centers. Controls uniform spacing.
    """
    style = get_panel_style() if style is None else style
    fontweight = "bold" if bold_fonts else "normal"

    df_main = _copy_with_lowercase_columns(df_main)
    df_amb = _copy_with_lowercase_columns(df_amb)

    if effect_cols_main is None:
        effect_cols_main = [
            "toxic",
            "hemolytic",
            "cytotoxic",
            "neurotoxic",
            "cytolysis",
            "embryotoxic",
            "ichthyotoxic",
        ]
    else:
        effect_cols_main = _normalize_names(effect_cols_main)

    if effect_cols_amb is None:
        effect_cols_amb = ["toxic", "hemolytic", "cytotoxic", "neurotoxic"]
    else:
        effect_cols_amb = _normalize_names(effect_cols_amb)

    if order is None:
        order = [
            "toxic",
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
    effect_cols_main = [
        endpoint for endpoint in effect_cols_main if endpoint not in excluded_endpoints
    ]
    effect_cols_amb = [
        endpoint for endpoint in effect_cols_amb if endpoint not in excluded_endpoints
    ]

    if endpoint_colors is None:
        endpoint_colors = {
            "toxic": "#CDA0B0",
            "cytotoxic": "#DFD8E7",
            "neurotoxic": "#DFD8E7",
            "embryotoxic": "#DFD8E7",
            "ichthyotoxic": "#DFD8E7",
            "hemolytic": "#FDE8D7",
            "cytolysis": "#FDE8D7",
        }
    else:
        endpoint_colors = {str(k).lower(): v for k, v in endpoint_colors.items()}

    if endpoint_label_map is None:
        endpoint_label_map = {
            "toxic": "Toxic",
            "cytotoxic": "Cytotoxic",
            "embryotoxic": "Embryotoxic",
            "ichthyotoxic": "Ichthyotoxic",
            "neurotoxic": "Neurotoxic",
            "hemolytic": "Hemolytic",
            "cytolysis": "Cytolysis",
            "cytolytic": "Cytolytic",
        }
    else:
        endpoint_label_map = {
            str(k).lower(): v for k, v in endpoint_label_map.items()
        }

    status_label_map = {
        "consistent negatives": "Consistent negatives",
        "consistent positives": "Consistent positives",
        "ambiguous": "Ambiguous",
    }

    _validate_required_columns(df_main, effect_cols_main, "df_main")
    _validate_required_columns(df_amb, effect_cols_amb, "df_amb")

    positives = (df_main[effect_cols_main] == 1).sum()
    negatives = (df_main[effect_cols_main] == 0).sum()

    ambiguous = pd.Series(
        {
            col: df_amb[col].apply(_is_ambiguous_value).sum()
            for col in effect_cols_amb
        }
    )

    summary = pd.DataFrame(
        {
            "consistent negatives": negatives,
            "consistent positives": positives,
            "ambiguous": ambiguous,
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
            f"The following endpoints have a total count equal to zero "
            f"and cannot be converted to percentages: {zero_rows}"
        )

    summary_pct = summary.div(row_totals, axis=0) * 100.0

    plot_order = [
        "consistent negatives",
        "consistent positives",
        "ambiguous",
    ]

    symbol_map = {
        "consistent negatives": "−",
        "consistent positives": "+",
        "ambiguous": "?",
    }

    def _format_percent(value: float) -> str:
        if abs(value - round(value)) < 0.05:
            return f"{value:.0f}%"
        return f"{value:.1f}%"

    fig, ax = plt.subplots(figsize=figsize)

    min_label_width = 7.0

    # Fixed y positions to keep equal spacing between all bars.
    y_positions = [
        row_idx * row_spacing
        for row_idx in range(len(summary_pct.index))
    ]

    for row_idx, endpoint in enumerate(summary_pct.index):
        y = y_positions[row_idx]

        base = endpoint_colors.get(endpoint, "#BFBFBF")

        color_map = {
            "consistent negatives": _darken_color(base, amount=0.72),
            "consistent positives": _darken_color(base, amount=0.88),
            "ambiguous": _lighten_color(base, amount=0.1),
        }

        left_val = 0.0

        for label_type in plot_order:
            value = float(summary_pct.loc[endpoint, label_type])

            if value <= 0:
                continue

            ax.barh(
                y,
                value,
                left=left_val,
                color=color_map[label_type],
                edgecolor="white",
                linewidth=0.85,
                height=bar_height,
            )

            # Optional: hide labels below a chosen threshold.
            # Example: hide_label_below_pct=1.0 hides labels below 1%.
            if hide_label_below_pct is not None and value < hide_label_below_pct:
                left_val += value
                continue

            label_text = f"{symbol_map[label_type]} {_format_percent(value)}"

            # Small segments: place label outside, in gray, with a guide line.
            if value < small_label_threshold:
                x_anchor = left_val + value
                x_text = x_anchor + outside_label_offset

                ax.annotate(
                    label_text,
                    xy=(x_anchor, y),
                    xytext=(x_text, y),
                    ha="left",
                    va="center",
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

                left_val += value
                continue

            # Larger segments: place label inside.
            ax.text(
                left_val + value / 2.0,
                y,
                label_text,
                ha="center",
                va="center",
                fontsize=annot_fontsize,
                color="white",
                alpha=0.92,
                fontweight="bold",
            )

            left_val += value

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            endpoint_label_map.get(endpoint, str(endpoint).capitalize())
            for endpoint in summary_pct.index
        ],
        fontsize=tick_fontsize,
        color=style["text_main"],
        fontweight=fontweight,
    )

    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))

    ax.set_xlabel(
        xlabel,
        fontsize=label_fontsize,
        color=style["text_main"],
        fontweight=fontweight,
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=title_fontsize,
            color=style["text_main"],
            pad=15,
            loc="left",
            fontweight=fontweight,
        )

    style_axes(ax, style, grid_axis="x")

    ax.tick_params(
        axis="x",
        labelsize=tick_fontsize,
        colors=style["text_main"],
    )

    ax.tick_params(
        axis="y",
        labelsize=tick_fontsize,
        length=0,
        colors=style["text_main"],
    )

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(fontweight)

    # Keeps the first endpoint at the top and preserves equal margins.
    if len(y_positions) > 0:
        ax.set_ylim(
            max(y_positions) + row_spacing * 0.55,
            -row_spacing * 0.55,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker=r"$+$",
            markersize=legend_marker_size,
            color="#6F7C89",
            label=status_label_map["consistent positives"],
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker=r"$-$",
            markersize=legend_marker_size,
            color="#6F7C89",
            label=status_label_map["consistent negatives"],
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker=r"$?$",
            markersize=legend_marker_size,
            color="#6F7C89",
            label=status_label_map["ambiguous"],
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
        ncol=3,
    )

    style_legend(legend, style)

    legend.get_title().set_fontsize(legend_title_fontsize)
    legend.get_title().set_fontweight(fontweight)

    for text in legend.get_texts():
        text.set_fontsize(legend_fontsize)
        text.set_fontweight(fontweight)

    fig.tight_layout()

    # Leaves space on the right for labels placed outside the bar.
    fig.subplots_adjust(right=0.88)

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
    Compute toxicity endpoint co-occurrence matrix and multi-endpoint statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing one binary column per toxicity endpoint.

    effect_order : list[str] | None
        Ordered list of toxicity endpoint columns to include.
        If None, a default toxicity endpoint order is used.

    positive_value : int | float | str
        Value used to define positive annotations. Default is 1.

    alias_map : dict[str, list[str]] | None
        Dictionary used to create compatible endpoint names from aliases.
        Example: {"cytolysis": ["cytolytic"]}

    verbose : bool
        If True, print summary information.

    Returns
    -------
    dict
        Dictionary containing:
        - effect_binary
        - row_totals
        - n_positive_endpoints_per_sequence
        - annotated_mask
        - multi_endpoint_mask
        - n_annotated_peptides
        - n_multi_endpoint_peptides
        - pct_multi_endpoint_peptides
        - cooccurrence_matrix
        - summary
    """

    if effect_order is None:
        effect_order = [
            "cytotoxic",
            "embryotoxic",
            "ichthyotoxic",
            "neurotoxic",
            "hemolytic",
            "cytolysis",
        ]

    effect_order = [effect.lower() for effect in effect_order]

    if alias_map is None:
        alias_map = {
            "cytolysis": ["cytolytic"],
        }

    df_tmp = _copy_with_lowercase_columns(df)

    # Create compatible columns from aliases when needed
    for target_col, aliases in alias_map.items():
        target_col = target_col.lower()

        if target_col not in df_tmp.columns:
            for alias in aliases:
                alias = alias.lower()
                if alias in df_tmp.columns:
                    df_tmp[target_col] = df_tmp[alias]
                    break

    missing_cols = [col for col in effect_order if col not in df_tmp.columns]

    if missing_cols:
        raise ValueError(f"Missing effect columns in dataframe: {missing_cols}")

    # Binary matrix: rows = sequences, columns = toxicity endpoints
    effect_binary = (df_tmp[effect_order] == positive_value).astype(int)

    # Number of positive sequences per endpoint
    row_totals = effect_binary.sum(axis=0)

    # Number of positive toxicity endpoints assigned to each peptide sequence
    n_positive_endpoints_per_sequence = effect_binary.sum(axis=1)

    # Peptides with at least one positive toxicity endpoint
    annotated_mask = n_positive_endpoints_per_sequence > 0

    # Peptides associated with more than one toxicity category
    multi_endpoint_mask = n_positive_endpoints_per_sequence > 1

    n_annotated_peptides = int(annotated_mask.sum())
    n_multi_endpoint_peptides = int(multi_endpoint_mask.sum())

    pct_multi_endpoint_peptides = (
        n_multi_endpoint_peptides / n_annotated_peptides * 100
        if n_annotated_peptides > 0
        else 0
    )

    # Co-occurrence matrix: endpoint-by-endpoint positive overlap counts
    cooccurrence_matrix = effect_binary.T @ effect_binary

    summary = {
        "n_unique_sequences": int(df_tmp.shape[0]),
        "n_annotated_peptides": n_annotated_peptides,
        "n_multi_endpoint_peptides": n_multi_endpoint_peptides,
        "pct_multi_endpoint_peptides": pct_multi_endpoint_peptides,
    }

    return {
        "effect_binary": effect_binary,
        "row_totals": row_totals,
        "n_positive_endpoints_per_sequence": n_positive_endpoints_per_sequence,
        "annotated_mask": annotated_mask,
        "multi_endpoint_mask": multi_endpoint_mask,
        "n_annotated_peptides": n_annotated_peptides,
        "n_multi_endpoint_peptides": n_multi_endpoint_peptides,
        "pct_multi_endpoint_peptides": pct_multi_endpoint_peptides,
        "cooccurrence_matrix": cooccurrence_matrix,
        "summary": summary,
    }