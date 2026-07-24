import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def plot_pairwise_similarity_panel(
    dfs: dict,
    toxic_effect: str,
    sim_thresholds: list[float],
    *,
    ncols: int = 4,
    bins: int = 100,
    kde: bool = True,
    figsize_per_panel: tuple[float, float] = (5, 4),
    legend_title: str = "Similarity thresholds",
    legend_nrows: int = 2,
    legend_bbox_to_anchor: tuple[float, float] = (0.5, -0.02),
    legend_loc: str = "upper center",
    save: bool = True,
    output_path: str | Path | None = None,
    dpi: int = 300,
):
    """
    Plot a panel of pairwise cosine similarity distributions, with one subplot
    per numerical representation.

    Parameters
    ----------
    dfs : dict
        Dictionary where keys are representation names and values are pandas
        DataFrames containing a column named 'similarity'.
    toxic_effect : str
        Name of the toxic effect to include in the figure title and file name.
    sim_thresholds : list[float]
        List of cosine similarity thresholds to draw as vertical dashed lines.
    ncols : int, default=4
        Number of subplot columns.
    bins : int, default=100
        Number of bins for the histogram.
    kde : bool, default=True
        Whether to overlay a KDE curve.
    figsize_per_panel : tuple[float, float], default=(5, 4)
        Size of each subplot as (width, height).
    legend_title : str, default="Similarity thresholds"
        Title of the global legend.
    legend_nrows : int, default=2
        Number of rows for the global legend.
    legend_bbox_to_anchor : tuple[float, float], default=(0.5, -0.02)
        Position of the global legend.
    legend_loc : str, default="upper center"
        Legend anchor location.
    save : bool, default=True
        Whether to save the figure.
    output_path : str | Path | None, default=None
        Output file path. If None, a default name is generated.
    dpi : int, default=300
        Resolution used when saving the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : np.ndarray
        Flattened array of subplot axes.
    """
    n_plots = len(dfs)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
    )

    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax, (name, df) in zip(axes, dfs.items()):
        sns.histplot(
            df["similarity"].values,
            bins=bins,
            kde=kde,
            ax=ax,
        )

        for thr in sim_thresholds:
            ax.axvline(
                thr,
                linestyle="--",
                linewidth=1,
                label=f"{thr:.3f}",
            )

        ax.set_title(name)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")

    # Remove empty subplots
    for i in range(len(dfs), len(axes)):
        fig.delaxes(axes[i])

    used_axes = axes[:len(dfs)]

    # Get legend handles/labels from the first used axis
    handles, labels = used_axes[0].get_legend_handles_labels()

    # Remove duplicated legend entries while preserving order
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)

    # Remove individual legends
    for ax in used_axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Number of columns to force the legend into legend_nrows rows
    legend_ncols = math.ceil(len(unique_labels) / legend_nrows)

    # Global legend below all subplots
    fig.legend(
        unique_handles,
        unique_labels,
        title=legend_title,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncols,
        frameon=True,
    )

    fig.suptitle(
        f"\nCosine similarity distributions {toxic_effect}",
        fontsize=16,
        y=0.98,
    )

    # Leave space at the bottom for the legend
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])

    if save:
        if output_path is None:
            output_path = f"pairwise_similarity_panel_{toxic_effect}.png"

        output_path = Path(output_path)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.show()


import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def plot_pairwise_distance_panel(
    dfs: dict,
    toxic_effect: str,
    dist_thresholds: list[float],
    *,
    ncols: int = 4,
    bins: int = 100,
    kde: bool = True,
    figsize_per_panel: tuple[float, float] = (5, 4),
    legend_title: str = "Distance thresholds",
    legend_nrows: int = 2,
    legend_bbox_to_anchor: tuple[float, float] = (0.5, -0.02),
    legend_loc: str = "upper center",
    save: bool = True,
    output_path: str | Path | None = None,
    dpi: int = 300,
):
    """
    Plot a panel of pairwise cosine distance distributions, with one subplot
    per numerical representation.
    """

    n_plots = len(dfs)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
    )

    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax, (name, df) in zip(axes, dfs.items()):
        sns.histplot(
            df["distance"].values,
            bins=bins,
            kde=kde,
            ax=ax,
        )

        for thr in dist_thresholds:
            ax.axvline(
                thr,
                linestyle="--",
                linewidth=1,
                label=f"{thr:.3f}",
            )

        ax.set_title(name)
        ax.set_xlabel("Cosine distance = 1 - cosine similarity")
        ax.set_ylabel("Count")

    # Remove empty subplots
    for i in range(len(dfs), len(axes)):
        fig.delaxes(axes[i])

    used_axes = axes[:len(dfs)]

    # Get legend handles/labels
    handles, labels = used_axes[0].get_legend_handles_labels()

    # Remove duplicates (MUY importante cuando hay muchas líneas)
    seen = set()
    unique_handles = []
    unique_labels = []

    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)

    # Remove subplot legends
    for ax in used_axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Calcular columnas para forzar 2 filas
    legend_ncols = math.ceil(len(unique_labels) / legend_nrows)

    # Global legend abajo
    fig.legend(
        unique_handles,
        unique_labels,
        title=legend_title,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncols,
        frameon=True,
    )

    fig.suptitle(
        f"\nCosine distance distributions {toxic_effect}",
        fontsize=16,
        y=0.98,
    )

    # Espacio para la leyenda
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])

    if save:
        if output_path is None:
            output_path = f"pairwise_distance_panel_{toxic_effect}.png"

        output_path = Path(output_path)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.show()