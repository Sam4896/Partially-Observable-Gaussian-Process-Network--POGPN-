import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from torch import scatter
import torch


def _setup_figure_axis(ax):
    """Set up figure and axis for 3D plotting."""
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()
    return fig, ax


def _prepare_grid_data(x, y, z, X_grid, Y_grid, Z_grid, method):
    """Prepare grid data either from scattered points or use provided grid."""
    if X_grid is not None and Y_grid is not None and Z_grid is not None:
        # Use pre-computed grid data directly
        return X_grid, Y_grid, Z_grid
    else:
        # Create grid from scattered points
        if x is None or y is None or z is None:
            raise ValueError(
                "Either provide (x, y, z) for scattered points or (X_grid, Y_grid, Z_grid) for grid data"
            )

        # Create a grid of points
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        X_grid, Y_grid = np.meshgrid(x_unique, y_unique)  # noqa: N806

        # Prepare points for griddata
        points = np.column_stack((x, y))
        values = z.flatten() if isinstance(z, np.ndarray) and z.ndim > 1 else z

        # Interpolate Z values onto the grid
        Z_grid = griddata(points, values, (X_grid, Y_grid), method=method)
        Z_grid = np.array(Z_grid).reshape(X_grid.shape)

        return X_grid, Y_grid, Z_grid


def _truncate_colormap(cmap):
    """Truncate colormap to avoid white colors for better visibility."""
    if isinstance(cmap, str) and cmap in [
        "Blues",
        "Reds",
        "Greens",
        "Purples",
        "Oranges",
        "YlOrRd",
    ]:
        original_cmap = mpl.cm.get_cmap(cmap)  # type: ignore
        truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
            f"trunc_{cmap}", original_cmap(np.linspace(0.25, 1.0, 256))
        )
        return truncated_cmap
    return cmap


def _plot_surface_with_contours(
    ax, X_grid, Y_grid, Z_grid, cmap, alpha, linewidth, antialiased, plot_contours
):
    """Plot the main surface and add contour projections."""

    Z_grid_max = 1.1 * np.max(Z_grid)
    # Plot the main surface
    surf = ax.plot_surface(
        X_grid,
        Y_grid,
        Z_grid,
        cmap=cmap,
        alpha=alpha,
        linewidth=linewidth,
        antialiased=antialiased,
    )

    if plot_contours:
        ax.contourf(
            X_grid,
            Y_grid,
            Z_grid,
            zdir="z",
            offset=Z_grid_max,
            levels=20,
            cmap=cmap,
            alpha=0.5,
        )

    return surf, Z_grid_max


def _add_scatter_points(ax, xs, ys, zs, color=None, label=None):
    """Add projected scatter points to the plot."""
    ax.scatter(
        xs=xs,
        ys=ys,
        zs=zs,
        color=color if color is not None else "black",
        s=50,
        alpha=0.3,
        label=label,
    )
    if label is not None:
        ax.legend()


def _finalize_plot(
    fig,
    ax,
    surf,
    x_label,
    y_label,
    z_label,
    colorbar_label,
    title,
    plot_colorbar=True,
):
    # Add colorbar
    if plot_colorbar:
        fig.colorbar(surf, shrink=0.5, aspect=5, label=colorbar_label)

    # Set labels and title
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)
    if title is not None:
        ax.set_title(title)


def plot_3d_surface(
    x=None,
    y=None,
    z=None,
    ax=None,
    X_grid=None,
    Y_grid=None,
    Z_grid=None,
    x_label=None,
    y_label=None,
    z_label=None,
    colorbar_label=None,
    title=None,
    cmap="Blues",
    alpha=0.3,
    method="cubic",
    linewidth=0,
    antialiased=False,
    scatter_points=None,
    scatter_points_color=None,
    scatter_points_label=None,
    plot_contours=True,
    plot_colorbar=True,
):
    """Plot 3D surface with optional colormap truncation.

    Can work with either:
    1. Scattered points (x, y, z) - will interpolate to grid
    2. Pre-computed grid data (X_grid, Y_grid, Z_grid) - will use directly

    Returns:
        tuple: (fig, ax, surf) where surf is the surface object for colorbars
    """
    # 1. Set up figure and axis
    fig, ax = _setup_figure_axis(ax)

    # 2. Prepare grid data
    X_grid, Y_grid, Z_grid = _prepare_grid_data(x, y, z, X_grid, Y_grid, Z_grid, method)

    # 3. Truncate colormap if needed
    cmap = _truncate_colormap(cmap)

    # 4. Plot surface with contours
    surf, Z_grid_max = _plot_surface_with_contours(
        ax, X_grid, Y_grid, Z_grid, cmap, alpha, linewidth, antialiased, plot_contours
    )

    # 5. Add scatter points if provided
    if scatter_points is not None:
        _add_scatter_points(
            ax,
            xs=scatter_points["x"].cpu().numpy(),
            ys=scatter_points["y"].cpu().numpy(),
            zs=scatter_points["z"].cpu().numpy(),
            color=scatter_points_color,
            label=scatter_points_label,
        )

    # 6. Finalize plot with colorbar, labels, and title
    _finalize_plot(
        fig, ax, surf, x_label, y_label, z_label, colorbar_label, title, plot_colorbar
    )

    return fig, ax, surf


def plot_all_surfaces(
    surface,
    data_dict,
    title_prefix,
    figsize=(13, 10),
    fontsize=9,
    titlesize=10,
):
    """Create a 2x2 subplot figure showing all surface plots."""
    # Create figure with adjusted size and spacing
    fig = plt.figure(figsize=figsize)

    # Add main title
    fig.suptitle(title_prefix.strip(": "), fontsize=titlesize, y=0.95)

    # Adjust subplot parameters for less whitespace
    plt.subplots_adjust(
        top=0.85,  # Make room for main title
        bottom=0.1,
        left=0.1,
        right=0.9,
        hspace=0.3,  # Reduce vertical spacing between subplots
        wspace=0.3,  # Reduce horizontal spacing between subplots
    )

    # Set smaller font size for all text elements
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": titlesize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
        }
    )

    # y1 vs x
    ax1 = fig.add_subplot(221, projection="3d")
    plot_3d_surface(
        x=surface["inputs"].cpu().numpy()[:, 0],
        y=surface["inputs"].cpu().numpy()[:, 1],
        z=surface["y1"].cpu().numpy(),
        x_label="X1",
        y_label="X2",
        z_label="Y1",
        colorbar_label="Y1",
        title="Y1 vs X",
        scatter_points={
            "x": data_dict["inputs"][:, 0],
            "y": data_dict["inputs"][:, 1],
            "z": torch.zeros_like(data_dict["y1"]) + torch.max(surface["y1"]),
        },
        scatter_points_color="black",
        scatter_points_label="Data points",
        ax=ax1,
    )

    # y2 vs x
    ax2 = fig.add_subplot(222, projection="3d")
    plot_3d_surface(
        x=surface["inputs"].cpu().numpy()[:, 0],
        y=surface["inputs"].cpu().numpy()[:, 1],
        z=surface["y2"].cpu().numpy(),
        x_label="X1",
        y_label="X2",
        z_label="Y2",
        colorbar_label="Y2",
        title="Y2 vs X",
        scatter_points={
            "x": data_dict["inputs"][:, 0],
            "y": data_dict["inputs"][:, 1],
            "z": torch.zeros_like(data_dict["y2"]) + torch.max(surface["y2"]),
        },
        scatter_points_color="black",
        scatter_points_label="Data points",
        ax=ax2,
    )

    # y3 vs x
    ax3 = fig.add_subplot(223, projection="3d")
    plot_3d_surface(
        x=surface["inputs"].cpu().numpy()[:, 0],
        y=surface["inputs"].cpu().numpy()[:, 1],
        z=surface["y3"].cpu().numpy(),
        x_label="X1",
        y_label="X2",
        z_label="Y3",
        colorbar_label="Y3",
        title="Y3 vs X",
        scatter_points={
            "x": data_dict["inputs"][:, 0],
            "y": data_dict["inputs"][:, 1],
            "z": torch.zeros_like(data_dict["y3"]) + torch.max(surface["y3"]),
        },
        scatter_points_color="black",
        scatter_points_label="Data points",
        ax=ax3,
    )

    # y3 vs y1, y2
    ax4 = fig.add_subplot(224, projection="3d")
    plot_3d_surface(
        x=surface["y1"].cpu().numpy(),
        y=surface["y2"].cpu().numpy(),
        z=surface["y3"].cpu().numpy(),
        x_label="Y1",
        y_label="Y2",
        z_label="Y3",
        colorbar_label="Y3",
        method="nearest",
        title="Y3 vs Y1, Y2",
        scatter_points={
            "x": data_dict["y1"],
            "y": data_dict["y2"],
            "z": torch.zeros_like(data_dict["y3"]) + torch.max(surface["y3"]),
        },
        scatter_points_color="black",
        scatter_points_label="Data points",
        ax=ax4,
    )

    plt.tight_layout()
    return fig, (ax1, ax2, ax3, ax4)
