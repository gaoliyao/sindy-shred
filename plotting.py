"""Plotting utilities for SINDy-SHRED visualization.

This module provides reusable plotting functions for visualizing:
- Latent space trajectories and SINDy model comparisons
- Spatial reconstruction comparisons
- Sensor time series predictions
- General pcolormesh utilities for spatio-temporal data

Functions
---------
plot_latent_comparison
    Compare SINDy-SHRED latent trajectories with identified model.
plot_reconstruction_comparison
    Compare real vs predicted spatial reconstructions.
plot_sensor_predictions
    Plot sensor time series in grid layout.
plot_pcolormesh
    Utility for consistent pcolormesh plotting.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_latent_comparison(
    gru_outs,
    x_sim,
    time=None,
    latent_dim=None,
    labels=("SINDy-SHRED", "Identified model"),
    figsize=None,
    title=None,
):
    """Plot comparison between SINDy-SHRED latent trajectories and SINDy-identified model.

    Parameters
    ----------
    gru_outs : array-like
        Latent space trajectories from SINDy-SHRED (n_samples, latent_dim).
    x_sim : array-like
        Simulated trajectories from identified SINDy model.
    time : array-like, optional
        Time array for x-axis. If None, uses sample indices.
    latent_dim : int, optional
        Number of latent dimensions to plot. If None, inferred from data.
    labels : tuple, optional
        Labels for the two trajectories.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Overall figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes or array of Axes
    """
    # Convert to numpy if needed
    if hasattr(gru_outs, "detach"):
        gru_outs = gru_outs.detach().cpu().numpy()

    if latent_dim is None:
        latent_dim = gru_outs.shape[1] if gru_outs.ndim > 1 else 1

    if time is None:
        time = np.arange(len(gru_outs))

    if figsize is None:
        figsize = (8, 2 * latent_dim)

    fig, axes = plt.subplots(latent_dim, 1, figsize=figsize, sharex=True, sharey=True)
    if latent_dim == 1:
        axes = [axes]

    for i in range(latent_dim):
        axes[i].plot(time, gru_outs[:, i], label=labels[0])
        axes[i].plot(time, x_sim[:, i], "k--", label=labels[1])
        axes[i].set_ylabel(rf"$z_{{{i}}}$ (-)")
        if i == latent_dim - 1:
            axes[i].set_xlabel("Time")
            axes[i].legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def plot_reconstruction_comparison(
    real_data,
    predicted_data,
    timesteps,
    sst_locs=None,
    grid_shape=(180, 360),
    lat_range=None,
    lon_range=None,
    cmap="twilight",
    figsize=None,
    diff_scale=10,
):
    """Plot comparison between real and predicted spatial reconstructions.

    Parameters
    ----------
    real_data : array-like
        Ground truth data (n_samples, n_features).
    predicted_data : array-like
        Predicted/reconstructed data.
    timesteps : list
        List of timestep indices to visualize.
    sst_locs : array-like, optional
        Indices of valid spatial locations. If None, uses all locations.
    grid_shape : tuple, optional
        Shape of spatial grid for reshaping (lat, lon). Default is (180, 360).
    lat_range : tuple, optional
        Latitude range for zooming (start, end). Default is full range.
    lon_range : tuple, optional
        Longitude range for zooming (start, end). Default is full range.
    cmap : str, optional
        Colormap name. Default is "twilight".
    figsize : tuple, optional
        Figure size.
    diff_scale : float, optional
        Scaling factor for difference plot. Default is 10.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    if lat_range is None:
        lat_range = (0, grid_shape[0])
    if lon_range is None:
        lon_range = (0, grid_shape[1])

    num_plots = len(timesteps)
    if figsize is None:
        figsize = (5 * num_plots, 10)

    fig, axes = plt.subplots(3, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = axes.reshape(-1, 1)

    vmin = np.percentile(
        np.concatenate([real_data.flatten(), predicted_data.flatten()]), 1
    )
    vmax = np.percentile(
        np.concatenate([real_data.flatten(), predicted_data.flatten()]), 99
    )

    for i, t in enumerate(timesteps):
        # Real data
        if sst_locs is not None:
            real_sst = np.zeros(grid_shape[0] * grid_shape[1])
            real_sst[sst_locs] = real_data[t, :]
        else:
            real_sst = real_data[t, :]
        reshaped_real = real_sst.reshape(grid_shape)
        real_zoomed = reshaped_real[lat_range[0] : lat_range[1], lon_range[0] : lon_range[1]]
        axes[0, i].imshow(real_zoomed, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"Real Data (t={t})")
        axes[0, i].set_axis_off()

        # Predicted data
        if sst_locs is not None:
            pred_sst = np.zeros(grid_shape[0] * grid_shape[1])
            pred_sst[sst_locs] = predicted_data[t, :]
        else:
            pred_sst = predicted_data[t, :]
        reshaped_pred = pred_sst.reshape(grid_shape)
        pred_zoomed = reshaped_pred[lat_range[0] : lat_range[1], lon_range[0] : lon_range[1]]
        axes[1, i].imshow(pred_zoomed, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"Prediction (t={t})")
        axes[1, i].set_axis_off()

        # Difference
        diff_zoomed = diff_scale * np.square(real_zoomed - pred_zoomed)
        axes[2, i].imshow(diff_zoomed, aspect="auto", cmap="plasma")
        axes[2, i].set_title(f"Squared Error (t={t})")
        axes[2, i].set_axis_off()

    fig.tight_layout()
    return fig, axes


def plot_sensor_predictions(
    real_data,
    predicted_data,
    sensor_locations,
    sensor_indices,
    num_train=52,
    num_pred=250,
    rows=4,
    cols=4,
    figsize=None,
    save_path=None,
):
    """Plot real vs predicted sensor time series in a grid layout.

    Parameters
    ----------
    real_data : array-like
        Real sensor data (n_samples, n_features).
    predicted_data : array-like
        Predicted sensor data.
    sensor_locations : array-like
        Array of sensor location indices.
    sensor_indices : list
        Indices into sensor_locations to plot.
    num_train : int, optional
        Number of training samples (for vertical line). Default is 52.
    num_pred : int, optional
        Number of prediction samples. Default is 250.
    rows : int, optional
        Number of rows in grid. Default is 4.
    cols : int, optional
        Number of columns in grid. Default is 4.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    if figsize is None:
        figsize = (3 * cols, 2 * rows)

    num_sensors = len(sensor_indices)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, sensor_idx in enumerate(sensor_indices):
        if i >= rows * cols:
            break

        sensor = sensor_locations[sensor_idx]

        # Real data for training + prediction period
        sensor_real = real_data[: num_train + num_pred, sensor]

        # Prediction data
        sensor_pred = predicted_data[num_train : num_train + num_pred, sensor]

        axes[i].plot(
            np.arange(num_train + num_pred), sensor_real, color="blue", linewidth=2
        )
        axes[i].plot(
            np.arange(num_train, num_train + num_pred),
            sensor_pred,
            color="red",
            linestyle="--",
            linewidth=2,
        )
        axes[i].set_title(f"Sensor {sensor_idx}", fontsize=10)
        axes[i].axvline(x=num_train, color="gray", linestyle=":", linewidth=1)

    # Remove unused subplots
    for i in range(num_sensors, rows * cols):
        fig.delaxes(axes[i])

    fig.text(0.5, 0.0, "Time", ha="center", va="center", fontsize=12)

    # Legend
    lines = [
        plt.Line2D([0], [0], color="blue", lw=2),
        plt.Line2D([0], [0], color="red", linestyle="--", lw=2),
    ]
    fig.legend(
        lines, ["Real Data", "Prediction"], loc="upper right", fontsize=8, frameon=False
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    return fig, axes


def plot_pcolormesh(data, time=None, space=None, ax=None, **kwargs):
    """Utility function for consistent pcolormesh plotting.

    Parameters
    ----------
    data : array-like
        2D data array (space x time).
    time : array-like, optional
        Time coordinates for x-axis.
    space : array-like, optional
        Space coordinates for y-axis.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.
    **kwargs
        Additional arguments passed to pcolormesh.

    Returns
    -------
    ax : matplotlib.axes.Axes
    mesh : matplotlib.collections.QuadMesh
    """
    if ax is None:
        fig, ax = plt.subplots()

    if time is None:
        time = np.arange(data.shape[1] if data.ndim > 1 else len(data))
    if space is None:
        space = np.arange(data.shape[0])

    default_kwargs = {"cmap": "RdBu_r", "rasterized": True}
    default_kwargs.update(kwargs)

    mesh = ax.pcolormesh(time, space, data, **default_kwargs)

    return ax, mesh


def plot_training_loss(validation_errors, figsize=(8, 4)):
    """Plot training validation errors over epochs.

    Parameters
    ----------
    validation_errors : array-like
        Validation errors recorded during training.
    figsize : tuple, optional
        Figure size. Default is (8, 4).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    if hasattr(validation_errors, "detach"):
        validation_errors = validation_errors.detach().cpu().numpy()

    epochs = np.arange(1, len(validation_errors) + 1)
    ax.plot(epochs, validation_errors, "b-", linewidth=2)
    ax.set_xlabel("Epoch (x threshold interval)")
    ax.set_ylabel("Validation Error")
    ax.set_title("Training Progress")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_timeseries_comparison(
    real_data,
    predicted_data,
    timesteps,
    figsize=None,
):
    """Plot comparison between real and predicted 1D time series data.

    Parameters
    ----------
    real_data : array-like
        Ground truth data (n_samples, n_features).
    predicted_data : array-like
        Predicted/reconstructed data.
    timesteps : list
        List of timestep indices to visualize.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    num_plots = len(timesteps)
    n_features = real_data.shape[1]

    if figsize is None:
        figsize = (4 * num_plots, 4)

    fig, axes = plt.subplots(2, num_plots, figsize=figsize, sharex=True, sharey=True)
    if num_plots == 1:
        axes = axes.reshape(-1, 1)

    for i, t in enumerate(timesteps):
        # Real data
        axes[0, i].bar(range(n_features), real_data[t, :], color="steelblue", alpha=0.7)
        axes[0, i].set_title(f"Real (t={t})")
        if i == 0:
            axes[0, i].set_ylabel("Value")

        # Predicted data
        axes[1, i].bar(range(n_features), predicted_data[t, :], color="coral", alpha=0.7)
        axes[1, i].set_title(f"Predicted (t={t})")
        axes[1, i].set_xlabel("Feature")
        if i == 0:
            axes[1, i].set_ylabel("Value")

    fig.tight_layout()
    return fig, axes


def plot_spatiotemporal_1d(
    data,
    time=None,
    space=None,
    title=None,
    figsize=(10, 4),
    **kwargs
):
    """Plot 1D spatio-temporal data as a heatmap.

    Useful for visualizing toy data systems where the spatial dimension
    is 1D (e.g., mixed oscillator systems).

    Parameters
    ----------
    data : array-like
        2D data array, shape (n_time, n_space) or (n_space, n_time).
    time : array-like, optional
        Time coordinates for x-axis.
    space : array-like, optional
        Space coordinates for y-axis.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size. Default is (10, 4).
    **kwargs
        Additional arguments passed to pcolormesh.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Handle data orientation (expect space x time for pcolormesh)
    if data.shape[0] > data.shape[1]:
        # Assume time x space, transpose
        data = data.T

    if time is None:
        time = np.arange(data.shape[1])
    if space is None:
        space = np.arange(data.shape[0])

    default_kwargs = {
        "cmap": "RdBu_r",
        "rasterized": True,
        "vmin": -3,
        "vmax": 3,
    }
    default_kwargs.update(kwargs)

    mesh = ax.pcolormesh(time, space, data, **default_kwargs)
    ax.set_xlabel("Time")
    ax.set_ylabel("Space")
    if title:
        ax.set_title(title)

    fig.colorbar(mesh, ax=ax)
    fig.tight_layout()

    return fig, ax


def plot_comparison(
    real,
    predicted,
    timesteps,
    data_type="2d",
    **kwargs
):
    """Unified comparison plot for any data type.

    Parameters
    ----------
    real : array-like
        Ground truth data (n_samples, n_features).
    predicted : array-like
        Predicted/reconstructed data.
    timesteps : list
        List of timestep indices to visualize.
    data_type : str, optional
        "2d" for spatial grids (SST-like), "1d" for time series arrays.
        Default is "2d".
    **kwargs
        Additional arguments passed to the underlying plot function.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    if data_type == "2d":
        return plot_reconstruction_comparison(real, predicted, timesteps, **kwargs)
    elif data_type == "1d":
        return plot_timeseries_comparison(real, predicted, timesteps, **kwargs)
    else:
        raise ValueError(f"data_type must be '2d' or '1d', got '{data_type}'")
