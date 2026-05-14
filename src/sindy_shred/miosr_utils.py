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
    t_train = np.arange(0, len(z) * shred._dt, shred._dt)

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
            fig, ax = plt.subplots(shred._latent_dim, sharex=True, sharey=True)
            if shred._latent_dim == 1:
                ax = [ax]
            for i in range(shred._latent_dim):
                ax[i].plot(
                    t_train,
                    z[:, i],
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
            fig, ax = plt.subplots(
                shred._latent_dim * 2, sharex=True, figsize=(8, 2 * shred._latent_dim)
            )
            if shred._latent_dim * 2 == 1:
                ax = [ax]
            # Plot positions
            for i in range(shred._latent_dim):
                ax[i].plot(t_train[1:-1], z_trimmed[:, i], label="SINDy-SHRED")
                ax[i].plot(t_train[1:-1], x_sim[:, i], "k--", label="identified model")
                ax[i].set_ylabel(rf"$z_{{{i}}}$ (-)")
                if i == 0:
                    ax[i].legend()
            # Plot velocities
            for i in range(shred._latent_dim):
                ax[shred._latent_dim + i].plot(
                    t_train[1:-1], dz[:, i], label="estimated"
                )
                ax[shred._latent_dim + i].plot(
                    t_train[1:-1],
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


def auto_tune_threshold(
    shred,
    thresholds=None,
    metric="sparsity_stable",
    test_steps=None,
    divergence_threshold=1e6,
    verbose=None,
    adaptive=True,
    scale_factor=0.3,
    n_thresholds=None,
    optimizer=None,
    optimizer_kwargs=None,
    differentiation_method=None,
):
    """Automatically select SINDy threshold via model evaluation.

    This method sweeps through candidate thresholds, fits a SINDy model for each,
    integrates forward to check for stability, and selects the best model based
    on the specified metric.

    Parameters
    ----------
    thresholds : array-like, optional
        Candidate threshold values to test. If None and adaptive=False, uses a
        default range. Ignored if adaptive=True.
    metric : str, optional
        Selection criterion. Options:
        - "sparsity_stable": Sparsest stable model (default)
        - "bic": Lowest Bayesian Information Criterion
        - "aic": Lowest Akaike Information Criterion
    test_steps : int, optional
        Number of steps for forward integration test. If None, uses test length.
    divergence_threshold : float, optional
        Max allowed value before model is considered divergent. Default is 1e6.
    verbose : bool, optional
        Print progress. If None, uses class verbose setting.
    adaptive : bool, optional
        If True, uses a nonparametric approach to determine thresholds:
        1. First fits a least-squares solution (threshold=0)
        2. Computes max_threshold = scale_factor * max(|coefficients|)
        3. Generates n_thresholds evenly spaced values from 0 to max_threshold
        Default is True.
    scale_factor : float, optional
        Fraction of max coefficient magnitude to use as max threshold when
        adaptive=True. Default is 0.3.
    n_thresholds : int, optional
        Number of threshold values to test when adaptive=True. Default is 10.
    optimizer : callable, optional
        Function to use in SINDy step. Default is STLSQ.
    optimizer_kwargs : dict, optional
        Arguments to pass to optimizer.
    differentiation_method : callable, optional
        Differentiation method to use in the SINDy step. Default is
        FiniteDifference.

    Returns
    -------
    best_threshold : float
        The selected threshold value.
    results : dict
        Dictionary with keys:
        - 'thresholds': tested threshold values
        - 'sparsity': number of nonzero coefficients for each
        - 'stable': whether each model is stable
        - 'bic': BIC for each (if computable)
        - 'aic': AIC for each (if computable)
        - 'best_idx': index of selected model
    """
    if verbose is None:
        verbose = shred._verbose

    if optimizer is None:
        raise ValueError("The MIOSR optimizer must be provided in the optimizer kwarg.")
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if "group_sparsity" in optimizer_kwargs:
        group_sparsity = optimizer_kwargs["group_sparsity"]
        sparsity = np.max(group_sparsity)
    else:
        # Pick an absurdly large sparsity.
        sparsity = 2 * shred._poly_order * shred._latent_dim * shred._ode_order
        group_sparsity = np.ones(shred._latent_dim) * sparsity
        if shred._ode_order == 2:
            group_sparsity = np.append(
                group_sparsity, np.ones(shred._latent_dim) * sparsity
            )

    if differentiation_method == "finite" or differentiation_method is None:
        shred._differentiation_method = ps.differentiation.FiniteDifference()
    elif differentiation_method == "smoothed finite":
        shred._differentiation_method = ps.differentiation.SmoothedFiniteDifference()

    if shred._gru_outs is None:
        # Need to get normalized latent space first
        gru_outs = shred.gru_normalize(data_type="train")
        x_train = gru_outs.detach().cpu().numpy()
    else:
        x_train = shred._gru_outs

    if shred._ode_order == 2:
        x_train, x_dot, z_trimmed, dz = shred._get_2nd_order_state(x_train)
    else:
        x_dot = None

    if adaptive:
        # Nonparametric approach: determine thresholds from least-squares solution
        if verbose:
            print("Computing least-squares solution to determine threshold range...")

        # Fit with threshold=0 to get the full least-squares solution
        miosr_model = ps.SINDy(
            optimizer=optimizer(group_sparsity=group_sparsity, **optimizer_kwargs),
            differentiation_method=shred._differentiation_method,
            feature_library=ps.PolynomialLibrary(degree=shred._poly_order),
        )
        miosr_model.fit(x_train, t=shred._dt, x_dot=x_dot)

        # Get max absolute coefficient value
        # coeffs = miosr_model.coefficients()
        # max_coeff = np.max(np.abs(coeffs))
        # max_threshold = scale_factor * max_coeff

        # Generate evenly spaced thresholds from 0 to max_threshold
        # thresholds = np.linspace(0, max_threshold, n_thresholds)

        if n_thresholds is None:
            # Either 10 steps or integer increments
            n_thresholds = int(np.max([(sparsity * 2 - sparsity / 2) // 10, 1]))
        if thresholds is None:
            thresholds = np.arange(sparsity / 2, sparsity * 2, n_thresholds)

        # if verbose:
        #     print(f"Max |coefficient|: {max_coeff:.4f}")
        #     print(f"Max threshold (scale_factor={scale_factor}): {max_threshold:.4f}")
        #     print(f"Testing {n_thresholds} thresholds: {thresholds}")

    if test_steps is None:
        test_steps = shred._train_length if shred._train_length else 100

    results = {
        "thresholds": thresholds,
        "sparsity": [],
        "stable": [],
        "bic": [],
        "aic": [],
        "mse": [],
    }

    for thresh in thresholds:
        if verbose:
            print(f"Testing threshold={thresh:.3f}...", end=" ")

        group_sparsity = np.ones(shred._latent_dim) * thresh
        if shred._ode_order == 2:
            group_sparsity = np.append(
                group_sparsity, np.ones(shred._latent_dim) * thresh
            )

        # Fit SINDy with this threshold
        model = ps.SINDy(
            optimizer=optimizer(group_sparsity=group_sparsity, **optimizer_kwargs),
            differentiation_method=shred._differentiation_method,
            feature_library=ps.PolynomialLibrary(degree=shred._poly_order),
        )
        model.fit(x_train, t=shred._dt, x_dot=x_dot)

        # Count nonzero coefficients (sparsity)
        n_nonzero = np.count_nonzero(model.coefficients())
        results["sparsity"].append(n_nonzero)

        # Test stability via forward integration
        try:
            t_test = np.arange(0, test_steps * shred._dt, shred._dt)
            init_cond = x_train[0, :]
            x_sim = model.simulate(init_cond, t_test)

            # Check for divergence
            if np.any(np.abs(x_sim) > divergence_threshold) or np.any(np.isnan(x_sim)):
                is_stable = False
                mse = np.inf
            else:
                is_stable = True
                # Compute MSE against test data
                n_compare = min(len(x_sim), len(x_train))
                mse = np.mean((x_sim[:n_compare] - x_train[:n_compare]) ** 2)
        except Exception:
            is_stable = False
            mse = np.inf

        results["stable"].append(is_stable)
        results["mse"].append(mse)

        # Compute information criteria (simplified)
        n_samples = len(x_train)
        n_params = n_nonzero
        if mse > 0 and mse < np.inf:
            # Approximate log-likelihood
            log_likelihood = -n_samples * np.log(mse + 1e-10) / 2
            bic = n_params * np.log(n_samples) - 2 * log_likelihood
            aic = 2 * n_params - 2 * log_likelihood
        else:
            bic = np.inf
            aic = np.inf

        results["bic"].append(bic)
        results["aic"].append(aic)

        if verbose:
            status = "stable" if is_stable else "DIVERGED"
            print(f"sparsity={n_nonzero}, {status}, MSE={mse:.4e}")

    # Select best model based on metric
    results["sparsity"] = np.array(results["sparsity"])
    results["stable"] = np.array(results["stable"])
    results["bic"] = np.array(results["bic"])
    results["aic"] = np.array(results["aic"])
    results["mse"] = np.array(results["mse"])

    stable_mask = results["stable"]

    if not np.any(stable_mask):
        if verbose:
            print("Warning: No stable models found. Using least divergent.")
        best_idx = np.argmin(results["mse"])
    elif metric == "sparsity_stable":
        # Among stable models, pick the sparsest
        stable_indices = np.where(stable_mask)[0]
        best_idx = stable_indices[np.argmin(results["sparsity"][stable_mask])]
    elif metric == "bic":
        stable_indices = np.where(stable_mask)[0]
        best_idx = stable_indices[np.argmin(results["bic"][stable_mask])]
    elif metric == "aic":
        stable_indices = np.where(stable_mask)[0]
        best_idx = stable_indices[np.argmin(results["aic"][stable_mask])]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    results["best_idx"] = best_idx
    best_threshold = thresholds[best_idx]

    if verbose:
        print(
            f"\nBest threshold: {best_threshold:.3f} "
            f"(sparsity={results['sparsity'][best_idx]}, "
            f"MSE={results['mse'][best_idx]:.4e})"
        )

    # Re-fit with the best threshold and store
    shred.sindy_identify(threshold=best_threshold, plot_result=False)

    return best_threshold, results
