import pysindy as ps
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", UserWarning)


def sindy_identify(
    shred,
    optimizer=None,
    optimizer_kwargs=None,
    differentiation_method=None,
    plot_result=True,
    save_path=None,
):
    """Post-hoc model discovery with SINDy using SHRED latent space trajectories.

    For 1st order systems (ode_order=1): Discovers z' = f(z)
    For 2nd order systems (ode_order=2): Discovers z'' = f(z, z')

    :param threshold: Sparsity threshold for SINDy
    :type threshold: float
    :param optimizer: Function to use in SINDy step. Default is STLSQ.
    :type optimizer: callable
    :param optimizer_kwargs: Keyword arguments to pass to the optimizer function.
    :type optimizer_kwargs: dict
    :param differentiation_method: Diff. method for SINDy.
    :type differentiation_method: str
    :param plot_result: Flag for plotting discovered model
    :type plot_result: bool
    :param save_path: Path to save the plot (without extension). If provided,
        saves both .pdf and .png versions. Example: "results/latent_comparison"
    :type save_path: str or None
    """

    if differentiation_method == "finite" or differentiation_method is None:
        shred._differentiation_method = ps.differentiation.FiniteDifference()
    elif differentiation_method == "smoothed finite":
        shred._differentiation_method = ps.differentiation.SmoothedFiniteDifference()

    if shred._train_data is None:
        raise ValueError("You need to call `fit` prior to recovering SINDy states.")

    # Normalized SINDy-SHRED latent space trajectories for post-hoc model discovery
    gru_outs = shred.gru_normalize(data_type="train")
    z = gru_outs.detach().cpu().numpy()
    shred._gru_outs = z

    if shred._ode_order == 1:
        # 1st order ODE: z' = f(z)
        model = ps.SINDy(
            optimizer=optimizer(**optimizer_kwargs),
            differentiation_method=shred._differentiation_method,
            feature_library=ps.PolynomialLibrary(degree=shred._poly_order),
        )
        model.fit(z, t=shred._dt)
        shred._model = model

        if shred._verbose:
            print("SINDy-derived dynamical equation (1st order):\n")
            model.print()

        # Plot the discovered SINDy model
        if plot_result or save_path:
            shred.sindy_simulate(z)
            x_sim = shred._x_sim
            t_train = np.arange(0, len(z) * shred._dt, shred._dt)
            fig, ax = plt.subplots(shred._latent_dim, sharex=True, sharey=True)
            if shred._latent_dim == 1:
                ax = [ax]
            for i in range(shred._latent_dim):
                ax[i].plot(
                    t_train,
                    gru_outs[:, i].detach().cpu().numpy(),
                    label="SINDy-SHRED",
                )
                ax[i].plot(t_train, x_sim[:, i], "k--", label="identified model")
                ax[i].set_ylabel(rf"$z_{{{i}}}$ (-)")
                if i == shred._latent_dim - 1:
                    ax[i].set_xlabel("time (n steps)")
                    ax[i].legend()
            if save_path:
                fig.savefig(f"{save_path}.pdf", bbox_inches="tight", dpi=300)
                fig.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
            if plot_result:
                plt.show()
            else:
                plt.close(fig)

    elif shred._ode_order == 2:
        x_state_trimmed, x_dot, z_trimmed, dz = shred._get_2nd_order_state(z)

        # Fit SINDy model with explicit derivatives
        model = ps.SINDy(
            optimizer=optimizer(**optimizer_kwargs),
            differentiation_method=differentiation_method,
            feature_library=ps.PolynomialLibrary(degree=shred._poly_order),
        )
        model.fit(x_state_trimmed, t=shred._dt, x_dot=x_dot)
        shred._model = model

        if shred._verbose:
            print("SINDy-derived dynamical equation (2nd order, state-space form):\n")
            print("State: [z_0, ..., z_{n-1}, v_0, ..., v_{n-1}]")
            print("where v = dz/dt\n")
            model.print()

        # Plot the discovered SINDy model
        if plot_result or save_path:
            shred.sindy_simulate(z)
            x_sim = shred._x_sim
            t_train = np.arange(0, len(z_trimmed) * shred._dt, shred._dt)
            fig, ax = plt.subplots(
                shred._latent_dim * 2, sharex=True, figsize=(8, 2 * shred._latent_dim)
            )
            if shred._latent_dim * 2 == 1:
                ax = [ax]
            # Plot positions
            for i in range(shred._latent_dim):
                ax[i].plot(t_train, z_trimmed[:, i], label="SINDy-SHRED")
                ax[i].plot(t_train, x_sim[:, i], "k--", label="identified model")
                ax[i].set_ylabel(rf"$z_{{{i}}}$ (-)")
                if i == 0:
                    ax[i].legend()
            # Plot velocities
            for i in range(shred._latent_dim):
                ax[shred._latent_dim + i].plot(t_train, dz[:, i], label="estimated")
                ax[shred._latent_dim + i].plot(
                    t_train,
                    x_sim[:, shred._latent_dim + i],
                    "k--",
                    label="identified model",
                )
                ax[shred._latent_dim + i].set_ylabel(rf"$\dot{{z}}_{{{i}}}$ (-)")
            ax[-1].set_xlabel("time (n steps)")
            plt.tight_layout()
            if save_path:
                fig.savefig(f"{save_path}.pdf", bbox_inches="tight", dpi=300)
                fig.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
            if plot_result:
                plt.show()
            else:
                plt.close(fig)
    else:
        raise ValueError(f"ode_order must be 1 or 2, got {shred._ode_order}")
