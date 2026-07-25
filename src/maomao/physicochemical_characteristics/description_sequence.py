import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from io import StringIO
from typing import Union, Mapping, Any
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from maomao.parsing.integrated_dataset_utils import *
from roxy.eda.summary import build_report
from roxy.report import dataset_report_to_html


def plot_sequence_length_histogram(
    df: pd.DataFrame,
    bins: int = 30,
    figsize: tuple = (10, 6),
    title: str = "Distribution of sequence lengths"
) -> None:
    """
    Plot the distribution of peptide sequence lengths stratified by label.

    This function expects a DataFrame where each row corresponds to one peptide
    sequence and includes:
      - a precomputed sequence length column (e.g., `df["length"] = df["sequence"].str.len()`)
      - a `label` column used to color the histogram

    The plot uses a histogram (counts) and optionally overlays a KDE curve per class
    to visualize distribution shape differences.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least:
        - "length": integer sequence length
        - "label" : class label (string or numeric)
    bins : int, default=30
        Number of histogram bins.
    figsize : tuple, default=(10, 6)
        Matplotlib figure size.
    title : str, default="Distribution of sequence lengths"
        Title displayed on the plot.

    Returns
    -------
    None
        Displays the plot.
    """
    # Safety check: if df is missing or empty, do nothing.
    if df is None or df.empty:
        return

    # Configure seaborn style for readable plots.
    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=figsize)

    # Histogram of length, colored by label.
    # `common_norm=False` avoids normalizing across classes together (useful in imbalance).
    sns.histplot(
        data=df,
        x="length",
        hue="label",
        bins=bins,
        kde=True,
        stat="count",
        common_norm=False,
        alpha=0.6
    )

    plt.xlabel("Sequence length")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pca_projection(
    X: pd.DataFrame,
    y,
    n_components: int = 2,
    figsize: tuple = (10, 6),
    palette: Union[str, Mapping[Any, str]] = "deep",
    point_size: int = 50,
    alpha: float = 0.75,
    random_state: int = 42,
    title: str = "PCA projection of sequence-level features",
    output_folder: str | None = None,
    filename: str = "pca_projection.png",
    dpi: int = 300,
    # extras útiles
    label_order: list[Any] | None = None,
    default_color: str = "#9E9E9E",
) -> None:
    """
    Project a feature matrix into 2D using PCA and visualize class separation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows = sequences, columns = descriptors).
    y : array-like
        Labels aligned with X rows.
    n_components : int, default=2
        Number of PCA components (typically 2 for visualization).
    figsize : tuple, default=(10, 6)
        Figure size.
    palette : str or dict, default="deep"
        If str: seaborn palette name (e.g., "deep").
        If dict: mapping {label_value: color_hex}. Example:
            {"No xxx evidence": "#4C78A8", "efecto-actividad": "#E45756"}
    point_size : int, default=50
        Scatter point size.
    alpha : float, default=0.75
        Point transparency.
    random_state : int, default=42
        Random seed for PCA.
    title : str
        Plot title.
    output_folder : str or None
        If provided, the figure is saved to this directory.
    filename : str
        Output filename when saving. If no extension is provided, ".png" is added.
    dpi : int
        Resolution for saved figure.
    label_order : list or None
        Optional explicit order for labels in plotting + legend.
    default_color : str
        Used only when palette is dict and a label is missing in the mapping.

    Returns
    -------
    None
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    labels = np.asarray(y)

    if len(labels) != len(X):
        raise ValueError(f"X and y must have the same number of rows. Got len(X)={len(X)}, len(y)={len(labels)}")

    # Fill NaNs to ensure PCA can run without errors.
    X_for_pca = X.fillna(0.0)

    # Fit PCA and project features.
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_for_pca)

    # Determine plotting order for labels
    if label_order is not None:
        unique_labels = list(label_order)
    else:
        unique_labels = list(pd.unique(labels))  # mantiene el orden de aparición

    # Build color mapping
    if isinstance(palette, dict):
        color_map = {k: v for k, v in palette.items()}
        colors_for_labels = {lab: color_map.get(lab, default_color) for lab in unique_labels}
    else:
        pal = sns.color_palette(palette, n_colors=len(unique_labels))
        colors_for_labels = {lab: pal[i] for i, lab in enumerate(unique_labels)}

    plt.figure(figsize=figsize)

    # Plot each class separately to keep legend clean and consistent.
    for lab in unique_labels:
        mask = labels == lab
        if not np.any(mask):
            continue
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=point_size,
            alpha=alpha,
            color=colors_for_labels[lab],
            edgecolor="black",
            linewidth=0.4,
            label=str(lab),
        )

    # Add explained variance to axis labels for interpretability.
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    plt.xlabel(f"PC1 ({pc1_var:.1f}%)", fontsize=12)
    plt.ylabel(f"PC2 ({pc2_var:.1f}%)", fontsize=12)
    plt.title(title, fontsize=14)

    plt.legend(title="Label", frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.4)
    sns.despine()
    plt.tight_layout()

    # Ensure file extension
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        if not os.path.splitext(filename)[1]:
            filename = f"{filename}.png"
        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_aa_frag_barplot(
    X: pd.DataFrame,
    y,
    aa_prefix: str = "AA_",
    figsize: tuple = (10, 6),
    palette: Union[str, Mapping[Any, str]] = "deep",
    output_folder: str | None = None,
    title: str = "Comparison of amino acid composition",
    filename: str = "aa_frag_barplot.png",
    dpi: int = 300,
    label_order: list[Any] | None = None,
    default_color: str = "#9E9E9E",
) -> None:
    """
    Compare amino-acid composition features across labels using a bar plot.

    This function assumes amino-acid features exist in X with a fixed prefix
    (e.g., "aa_frac_" or "AA_"). It computes per-label mean composition for
    each residue feature and plots them as grouped bars.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing amino-acid composition columns.
    y : array-like
        Labels aligned with X rows.
    aa_prefix : str, default="AA_"
        Prefix used to identify amino-acid composition columns.
    figsize : tuple, default=(10, 6)
        Figure size.
    palette : str or dict, default="deep"
        Seaborn palette name OR dictionary mapping {label: color}.
        Example:
            {"No xxx evidence": "#4C78A8",
             "efecto-actividad": "#E45756"}
    output_folder : str or None, default=None
        If provided, the figure is saved to this directory.
    title : str
        Plot title.
    filename : str, default="aa_frag_barplot.png"
        Output filename when saving.
    dpi : int, default=300
        Resolution for saved figure.
    label_order : list or None
        Optional explicit order for legend and bar grouping.
    default_color : str
        Fallback color if a label is missing from palette dict.

    Returns
    -------
    None
        Displays the plot and optionally saves it to disk.
    """
    # Attach labels to the feature matrix for aggregation.
    df = X.copy()
    df["label"] = y

    # Identify amino-acid composition columns.
    aa_cols = [c for c in df.columns if c.startswith(aa_prefix)]
    if not aa_cols:
        return

    # Compute mean composition per label and reshape to long format for seaborn.
    mean_df = (
        df.groupby("label")[aa_cols]
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

    # Determine label order for plotting
    if label_order is not None:
        hue_order = label_order
    else:
        hue_order = list(mean_df["Label"].unique())

    # Build palette mapping
    if isinstance(palette, dict):
        palette_final = {
            lab: palette.get(lab, default_color)
            for lab in hue_order
        }
    else:
        colors = sns.color_palette(palette, n_colors=len(hue_order))
        palette_final = dict(zip(hue_order, colors))

    plt.figure(figsize=figsize)
    plt.grid(True, linestyle="--", alpha=0.4)

    sns.barplot(
        data=mean_df,
        x="Residue",
        y="AA_frag",
        hue="Label",
        hue_order=hue_order,
        palette=palette_final
    )

    plt.ylabel("AA fragment frequency")
    plt.xlabel("Residue")
    plt.xticks(rotation=90)
    plt.title(title)
    sns.despine()
    plt.tight_layout()

    # Save figure if requested.
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        if not os.path.splitext(filename)[1]:
            filename = f"{filename}.png"
        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.show()

def plot_descriptor_boxplots_panel(
    X: pd.DataFrame,
    y,
    exclude_prefix: str = "AA_",
    palette: Union[str, Mapping[Any, str]] = "deep",
    n_rows: int = 5,
    n_cols: int = 4,
    figsize: tuple = (16, 18),
    output_folder: str | None = None,
    filename: str = "descriptor_boxplots_panel.png",
    dpi: int = 300,
    label_order: list[Any] | None = None,
    default_color: str = "#9E9E9E",
) -> None:
    """
    Plot a grid (panel) of boxplots for numeric descriptors stratified by label.

    The panel is useful to quickly inspect distribution shifts across classes
    for multiple physicochemical descriptors. Amino-acid composition features
    can be excluded by prefix to focus on continuous descriptors.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows = sequences, columns = descriptors).
    y : array-like
        Labels aligned with X rows.
    exclude_prefix : str, default="AA_"
        Descriptor columns starting with this prefix are excluded (commonly AA composition).
    palette : str or dict, default="deep"
        Seaborn palette name OR dictionary mapping {label: color}.
    n_rows : int, default=5
        Number of subplot rows.
    n_cols : int, default=4
        Number of subplot columns.
    figsize : tuple, default=(16, 18)
        Figure size.
    output_folder : str or None, default=None
        If provided, the figure is saved to this directory.
    filename : str, default="descriptor_boxplots_panel.png"
        Output filename when saving.
    dpi : int, default=300
        Resolution for saved figure.
    label_order : list or None
        Optional explicit order for labels across all subplots.
    default_color : str
        Fallback color if a label is missing from palette dict.

    Returns
    -------
    None
        Displays the panel and optionally saves it to disk.
    """
    df = X.copy()
    df["label"] = y

    # Keep a consistent label order for all subplots.
    if label_order is not None:
        order = label_order
    else:
        order = sorted(df["label"].unique())

    # Build palette mapping (consistent with PCA and barplot)
    if isinstance(palette, dict):
        palette_final = {
            lab: palette.get(lab, default_color)
            for lab in order
        }
    else:
        colors = sns.color_palette(palette, n_colors=len(order))
        palette_final = dict(zip(order, colors))

    # Select numeric descriptor columns excluding AA composition and label,
    # and enforce alphabetical ordering.
    numeric_cols = sorted([
        c for c in df.select_dtypes(include=np.number).columns
        if not c.startswith(exclude_prefix) and c != "label"
    ])

    # Limit number of plots to available panel slots.
    max_plots = n_rows * n_cols
    numeric_cols = numeric_cols[:max_plots]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    # Create one boxplot per descriptor.
    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(
            data=df,
            x="label",
            y=col,
            order=order,
            hue="label",
            palette=palette_final,
            showfliers=False,
            legend=False,
            width=0.5,
            ax=ax,
            boxprops=dict(edgecolor="0.3", linewidth=1.1),
            whiskerprops=dict(color="0.3", linewidth=1.1),
            capprops=dict(color="0.3", linewidth=1.1),
            medianprops=dict(color="0.2", linewidth=1.4),
        )
        ax.set_title(col.replace("_", " "), fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Turn off any unused axes.
    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    sns.despine(trim=True)
    plt.tight_layout()

    # Save figure if requested.
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        if not os.path.splitext(filename)[1]:
            filename = f"{filename}.png"
        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.show()

def plot_descriptor_hist_kde_panel(
    X: pd.DataFrame,
    y,
    exclude_prefix: str = "AA_",
    palette: Union[str, Mapping[Any, str]] = "deep",
    n_rows: int = 5,
    n_cols: int = 4,
    figsize: tuple = (16, 18),
    bins: int = 30,
    kde: bool = True,
    stat: str = "density",
    common_norm: bool = False,
    element: str = "step",
    alpha: float = 0.35,
    linewidth: float = 1.2,
    output_folder: str | None = None,
    filename: str = "descriptor_hist_kde_panel.png",
    dpi: int = 300,
    legend_title: str = "Label",
    legend_loc: str = "upper center",
    bbox_to_anchor: tuple = (0.5, -0.05),
    label_order: list[Any] | None = None,
    default_color: str = "#9E9E9E",
) -> None:
    """
    Plot a panel of histograms (and optional KDE curves) per descriptor, stratified by label.

    This visualization is designed for comparing descriptor distribution *shapes*
    between classes, especially under class imbalance. By default, it uses:
      - stat="density" and common_norm=False to make distributions comparable across labels
      - element="step" and alpha<1.0 to improve readability in overlaps
      - a single global legend to avoid clutter in subplots

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with numeric descriptors.
    y : array-like
        Labels aligned with X rows.
    exclude_prefix : str, default="AA_"
        Exclude columns starting with this prefix (e.g., amino acid composition).
    palette : str or dict, default="deep"
        Seaborn palette name OR dictionary mapping {label: color}.
    n_rows : int, default=5
        Number of subplot rows.
    n_cols : int, default=4
        Number of subplot columns.
    figsize : tuple, default=(16, 18)
        Figure size.
    bins : int, default=30
        Number of histogram bins.
    kde : bool, default=True
        Whether to overlay KDE curves.
    stat : str, default="density"
        Histogram statistic ("count" or "density"). "density" is recommended for shape comparison.
    common_norm : bool, default=False
        If False, each label is normalized independently (recommended under imbalance).
    element : str, default="step"
        Histogram element style ("bars", "step", "poly").
    alpha : float, default=0.35
        Histogram transparency.
    linewidth : float, default=1.2
        Line width for step histograms and KDE curves.
    output_folder : str or None, default=None
        Output directory for saving the figure. If None, the figure is only shown.
    filename : str, default="descriptor_hist_kde_panel.png"
        Output filename for the saved figure.
    dpi : int, default=300
        Resolution for saved figure.
    legend_title : str, default="Label"
        Global legend title.
    legend_loc : str, default="upper center"
        Location of the global legend.
    bbox_to_anchor : tuple, default=(0.5, -0.05)
        Legend anchor position.
    label_order : list or None
        Optional explicit order for labels across all subplots/legend.
    default_color : str
        Fallback color if a label is missing from palette dict.

    Returns
    -------
    None
        Displays the panel and optionally saves it to disk.

    Notes
    -----
    - If `output_folder` is provided, the directory is created if missing.
    - This function constructs a manual legend to ensure consistent class-color mapping.
    """
    df = X.copy()
    df["label"] = y

    # Consistent label ordering across plots.
    if label_order is not None:
        order = label_order
    else:
        order = sorted(df["label"].dropna().unique())

    # Build an explicit color mapping (robust legend + consistent colors).
    if isinstance(palette, dict):
        label_to_color = {lab: palette.get(lab, default_color) for lab in order}
    else:
        palette_colors = sns.color_palette(palette, n_colors=len(order))
        label_to_color = dict(zip(order, palette_colors))

    # Select numeric descriptor columns excluding AA composition and label,
    # and enforce alphabetical ordering.
    numeric_cols = sorted([
        c for c in df.select_dtypes(include=np.number).columns
        if not c.startswith(exclude_prefix) and c != "label"
    ])

    max_plots = n_rows * n_cols
    numeric_cols = numeric_cols[:max_plots]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.ravel(axes)

    # One subplot per descriptor.
    for ax, col in zip(axes, numeric_cols):
        sns.histplot(
            data=df,
            x=col,
            hue="label",
            hue_order=order,
            bins=bins,
            stat=stat,
            common_norm=common_norm,
            element=element,
            alpha=alpha,
            palette=label_to_color,
            linewidth=linewidth,
            ax=ax,
            legend=False,  # avoid per-subplot legends
        )

        if kde:
            sns.kdeplot(
                data=df,
                x=col,
                hue="label",
                hue_order=order,
                common_norm=common_norm,
                palette=label_to_color,
                linewidth=linewidth,
                ax=ax,
                legend=False
            )

        ax.set_title(col.replace("_", " "), fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Disable unused axes.
    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    # Global legend (manual handles for stability).
    legend_handles = [
        Patch(facecolor=label_to_color[lab], edgecolor="black", label=str(lab))
        for lab in order
    ]

    fig.legend(
        handles=legend_handles,
        title=legend_title,
        loc=legend_loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=min(len(order), 4),
        frameon=True,
        fancybox=True,
        framealpha=0.9
    )

    sns.despine(trim=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save if requested.
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        if not os.path.splitext(filename)[1]:
            filename = f"{filename}.png"
        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_effect_distribution(
    df: pd.DataFrame,
    column: str = "effect",
    bins: int = 15,
    threshold: float = 0.15,
    figsize: tuple = (6, 4),
    title: str = "Effect size distribution of selected descriptors"
) -> None:
    """
    Plot the distribution of effect sizes (e.g., Cliff's delta) across descriptors.

    This utility is intended for summarizing results from statistical comparisons
    between classes (e.g., descriptor-level effect sizes). It draws:
      - a histogram with KDE
      - reference vertical lines at 0 and ±threshold to help interpret magnitude

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the effect size column.
    column : str, default="effect"
        Column name with effect sizes (e.g., Cliff's delta values).
    bins : int, default=15
        Number of histogram bins.
    threshold : float, default=0.15
        Reference threshold for "small" effects (adjust according to your criteria).
    figsize : tuple, default=(6, 4)
        Figure size.
    title : str, default="Effect size distribution of selected descriptors"
        Plot title.

    Returns
    -------
    None
        Displays the plot.
    """
    plt.figure(figsize=figsize)

    sns.histplot(df[column], bins=bins, kde=True)

    plt.axvline(0, color="black", linestyle="--", label="0")
    plt.axvline(threshold, color="red", linestyle=":", label=f"+{threshold}")
    plt.axvline(-threshold, color="red", linestyle=":", label=f"-{threshold}")

    plt.xlabel("Cliff's delta")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def generate_dataset_eda_report(
    X,
    y,
    activity: str,
    output_folder: str,
    max_features: int = 45,
    include_correlation: bool = True,
    task_type: str = "classification",
    filename: str = "roxy_dataset_report.html",
) -> str:
    """
    Generate a ROXY dataset-level EDA report and export it as an HTML file.

    The report summarizes dataset properties and feature behavior, typically including:
      - basic statistics (shape, missingness)
      - label distribution (for classification)
      - feature-level summaries (top features by signal/variance)
      - optional correlation analysis

    This function is designed to be used after computing sequence descriptors
    (e.g., via `RoxyHelpers.describe_sequences`) and provides a reproducible
    HTML artifact for dataset documentation and reporting.

    Parameters
    ----------
    X : pd.DataFrame or array-like
        Feature matrix (rows = sequences, columns = descriptors).
    y : pd.Series or array-like
        Labels aligned with X rows.
    activity : str
        Dataset name or task identifier used in the report title.
    output_folder : str
        Directory where the HTML report will be saved.
    max_features : int, default=45
        Maximum number of descriptors to include in the HTML report.
    include_correlation : bool, default=True
        Whether to include correlation analysis in the report.
    task_type : str, default="classification"
        Type of ML task ("classification" or "regression").
    filename : str, default="roxy_dataset_report.html"
        Output HTML filename.

    Returns
    -------
    str
        Full path to the saved HTML report.
    """
    # Ensure output directory exists.
    os.makedirs(output_folder, exist_ok=True)

    # Build the dataset-level report object.
    dataset_report = build_report(
        X,
        y=y,
        dataset_name=f"Description of the {activity} dataset",
        task_type=task_type,
    )

    # Convert the report object into HTML.
    html_report = dataset_report_to_html(
        dataset_report,
        max_features=max_features,
        include_correlation=include_correlation,
    )

    # Save HTML to disk.
    html_path = os.path.join(output_folder, filename)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)

    return html_path
