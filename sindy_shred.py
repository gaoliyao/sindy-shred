import torch
import sindy
import pysindy as ps
import sindy_shred_net
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import warnings

from utils import get_device, TimeSeriesDataset

warnings.simplefilter("ignore", UserWarning)


class SINDySHRED:
    """The infrastructure for fitting and working with a SINDy-SHRED model.

    :param latent_dim:
    :type latent_dim:
    :param poly_order:
    :type poly_order:
    :param include_sine:
    :type include_sine:
    :param hidden_layers:
    :type hidden_layers:
    :param l1:
    :type l1:
    :param l2:
    :type l2:
    :param dropout:
    :type dropout:
    :param batch_size:
    :type batch_size:
    :param num_epochs:
    :type num_epochs:
    :param lr:
    :type lr:
    :param verbose:
    :type verbose:
    :param threshold:
    :type threshold:
    :param patience:
    :type patience:
    :param sindy_regularization:
    :type sindy_regularization:
    :param optimizer:
    :type optimizer:
    :param thres_epoch:
    :type thres_epoch:
    :param device:
    :type device:
    :param sample_mode:
    :type sample_mode:
    """

    def __init__(
        self,
        latent_dim=None,
        poly_order=None,
        include_sine=False,
        ode_order=1,  # NEW: 1 for x' = f(x), 2 for x'' = f(x, x')
        hidden_layers=2,
        l1=350,
        l2=400,
        dropout=0.1,
        batch_size=128,  # How many trajectory time steps to train on in a single
        # gradient descent. The smaller it is, the more optimization steps it will take.
        num_epochs=200,
        lr=1e-3,
        verbose=True,
        threshold=0.25,  # Really large = SINDy won't activate, then it becomes SHRED
        patience=None,
        sindy_regularization=10.0,  # Set to 0 also becomes SHRED (Better than
        # threshold)
        optimizer="AdamW",
        thres_epoch=100,
        device=None,
        sample_mode=None,
    ):

        self._x_predict = None
        self._x_sim = None
        self._model = None
        self._shred = None
        self._device = device if device is not None else get_device()
        self._ode_order = ode_order

        self._train_data = None
        self._valid_data = None
        self._lags = None
        self._num_sensors = None
        self._n_space_dim = None
        self._n_time_dim = None

        self._hidden_layers = hidden_layers
        self._latent_dim = latent_dim
        self._poly_order = poly_order
        self._verbose = verbose
        self._sample_mode = sample_mode

        include_constant = True
        self._library_dim = sindy.library_size(
            self._latent_dim, self._poly_order, include_sine, include_constant
        )

        self._sindy_shred_kwargs = {
            "hidden_size": self._latent_dim,
            "hidden_layers": self._hidden_layers,
            "l1": l1,
            "l2": l2,
            "dropout": dropout,
            "library_dim": self._library_dim,
            "poly_order": self._poly_order,
            "include_sine": include_sine,
        }

        self._fit_kwargs = {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "verbose": self._verbose,
            "threshold": threshold,
            "patience": patience,
            "sindy_regularization": sindy_regularization,
            "optimizer": optimizer,
            "thres_epoch": thres_epoch,
        }

    @staticmethod
    def relative_error(x_est, x_true):
        """Helper function for calculating the relative error.

        :param x_est: Estimated values (i.e. from reconstruction)
        :type x_est: numpy.ndarray
        :param x_true: True (or observed) values.
        :type x_true: numpy.ndarray
        :return: Relative error between observations and model.
        :rtype: numpy.ndarray
        """
        return np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true)

    @staticmethod
    def estimate_velocity(z, dt):
        """Estimate dz/dt from latent trajectories using finite difference.

        For 2nd order ODE systems, we need velocity (z') in addition to position (z).
        This method estimates velocity using centered finite differences.

        Parameters
        ----------
        z : numpy.ndarray
            Position trajectories, shape (n_samples, latent_dim).
        dt : float
            Time step between samples.

        Returns
        -------
        dz : numpy.ndarray
            Estimated velocity, shape (n_samples-2, latent_dim) for centered diff,
            or (n_samples-1, latent_dim) for forward diff.
        z_trimmed : numpy.ndarray
            Position array trimmed to match velocity array length.
        """
        # Use centered finite difference for better accuracy
        # dz[i] = (z[i+1] - z[i-1]) / (2*dt)
        dz = (z[2:] - z[:-2]) / (2 * dt)
        z_trimmed = z[1:-1]  # Match the length of centered diff
        return dz, z_trimmed

    def _get_data_dims(self, x_to_fit):
        """Assign the data size properties"""
        self._n_time_dim, self._n_space_dim = x_to_fit.shape

    def _generate_splits(
        self,
        train_length,
        validate_length,
        lags,
        test_length=None,
        mode=None,
    ):
        """
        Generates train, validate, and test splits according to a specified strategy.

        :param train_length: Length of the training data split
        :type train_length: int
        :param validate_length: Length of the validation data split
        :type validate_length: int
        :param lags: Length of the sensor trajectories
        :type lags: int
        :param test_length: Length of the test data split
        :type test_length: int
        :param mode: Only the forecast data split is implemented.
        :type mode: str
        :param sample_size: Unused (should be part of the reconstruct mode)
        :type sample_size: int
        """
        if mode is None:
            mode = "forecast"

        if mode == "forecast":
            if test_length is None:
                test_length = self._n_time_dim - (train_length + validate_length + lags)

            train_ind = np.arange(0, train_length)

            mask = np.ones(self._n_time_dim - lags)
            mask[train_ind] = 0
            val_test_ind = np.arange(0, self._n_time_dim - lags)[np.where(mask != 0)[0]]
            val_ind = val_test_ind[:validate_length]

            test_ind = val_test_ind[validate_length : test_length + validate_length]

        elif mode == "reconstruct":
            # In this mode we do not withhold a test dataset unless directed otherwise.
            if test_length is None:
                test_length = 0

            train_ind = np.arange(0, train_length)

            mask = np.ones(self._n_time_dim - lags)
            mask[train_ind] = 0
            val_test_ind = np.arange(0, self._n_time_dim - lags)[np.where(mask != 0)[0]]
            # val_ind = val_test_ind[:validate_length]
            val_ind = val_test_ind
            # test_ind = val_test_ind[validate_length:test_length + validate_length]
            test_ind = None

            # train_ind = np.random.choice(
            #     self._n_time_dim - self._lags,
            #     size=train_length,
            #     replace=False
            # )
            # mask = np.ones(self._n_time_dim - lags)
            # mask[train_ind] = 0
            # valid_test_indices = np.arange(0, self._n_time_dim - self._lags)[np.where(
            #     mask != 0)[0]]
            # val_ind = valid_test_indices
            # val_ind = valid_test_indices[::2]
            # test_ind = valid_test_indices[1::2]
        else:
            raise ValueError(
                f"Unexpected mode (mode={mode}) provided. Valid options "
                f"are forecast and reconstruct."
            )

        self._train_ind = train_ind
        self._val_ind = val_ind
        self._test_ind = test_ind
        self._test_length = len(test_ind) if test_ind is not None else 0

    def _scale_data(self, x_to_fit):
        """Transform data and generate input/output pairs for each data split

        :param x_to_fit: Data to be transformed.
        :type x_to_fit: numpy.ndarray
        """
        # Scaling
        sc = MinMaxScaler()
        train_ind = self._train_ind
        val_ind = self._val_ind
        test_ind = self._test_ind
        lags = self._lags
        num_sensors = self._num_sensors
        sensor_locations = self._sensor_locations

        sc = sc.fit(x_to_fit[train_ind])
        transformed_X = sc.transform(x_to_fit)

        # Generate input sequences to a SHRED model
        all_data_in = np.zeros((self._n_time_dim - lags, lags, num_sensors))
        for i in range(len(all_data_in)):
            all_data_in[i] = transformed_X[i : i + lags, sensor_locations]

        # Generate training validation and test datasets both for reconstruction of
        # states and forecasting sensors
        train_data_in = torch.tensor(all_data_in[train_ind], dtype=torch.float32).to(
            self._device
        )
        valid_data_in = torch.tensor(all_data_in[val_ind], dtype=torch.float32).to(
            self._device
        )

        # -1 to have output be at the same time as final sensor measurements
        train_data_out = torch.tensor(
            transformed_X[train_ind + lags - 1], dtype=torch.float32
        ).to(self._device)
        valid_data_out = torch.tensor(
            transformed_X[val_ind + lags - 1], dtype=torch.float32
        ).to(self._device)

        train_data = TimeSeriesDataset(train_data_in, train_data_out)
        valid_data = TimeSeriesDataset(valid_data_in, valid_data_out)

        self._train_data = train_data
        self._valid_data = valid_data
        self._scaler = sc

        # Only scale test data if we have test data (i.e., not reconstructing)
        if self._test_ind is not None:
            test_data_in = torch.tensor(all_data_in[test_ind], dtype=torch.float32).to(
                self._device
            )
            test_data_out = torch.tensor(
                transformed_X[test_ind + lags - 1], dtype=torch.float32
            ).to(self._device)
            test_data = TimeSeriesDataset(test_data_in, test_data_out)
            self._test_data = test_data

    def fit(
        self,
        num_sensors,
        dt,
        x_to_fit,
        lags,
        train_length,
        validate_length,
        sensor_locations,
        test_length=None,
        seed=0,
    ):
        """Fit SINDy-SHRED to the given data.

        Performs the data split behind the scenes according to the given strategy and
        length of the data splits given by the user.

        :param num_sensors: Number of sensor trajectories. Used to initialize arrays.
        :type num_sensors: int
        :param dt: Time step of the data
        :type dt: float
        :param x_to_fit: Data to fit with SINDy-SHRED.
        :type x_to_fit: numpy.ndarray
        :param lags: Length of trajectories for discovering latent space
        :type lags: int
        :param train_length: Length of the training data
        :type train_length: int
        :param validate_length: Length of the validation data (warning, might be
        largely unused)
        :type validate_length: int
        :param sensor_locations: Locations of the data in the higher dimensional space
        :type sensor_locations: int
        :param test_length: Length of the test data. Only stored and transformed.
        :type test_length: int
        :param seed: Value for setting the random number seeds.
        :type seed: int
        """

        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self._dt = dt
        self._lags = lags
        self._train_length = train_length
        self._valid_length = validate_length
        self._test_length = test_length
        self._num_sensors = num_sensors
        self._sensor_locations = sensor_locations

        # Determine the data dimensions
        self._get_data_dims(x_to_fit)
        # Prepare the data for fitting by creating train, validation, and test splits
        self._generate_splits(
            self._train_length,
            self._valid_length,
            self._lags,
            test_length=self._test_length,
            mode=self._sample_mode,
        )
        # and then scale the data
        self._scale_data(x_to_fit)

        shred = sindy_shred_net.SINDy_SHRED_net(
            self._num_sensors,
            self._n_space_dim,
            dt=self._dt,
            **self._sindy_shred_kwargs,
        ).to(self._device)

        validation_errors = sindy_shred_net.fit(
            shred, self._train_data, self._valid_data, **self._fit_kwargs
        )

        self._shred = shred

    def sindy_identify(
        self,
        threshold,
        differentiation_method="finite",
        plot_result=True,
        save_path=None,
    ):
        """Post-hoc model discovery with SINDy using SHRED latent space trajectories.

        For 1st order systems (ode_order=1): Discovers z' = f(z)
        For 2nd order systems (ode_order=2): Discovers z'' = f(z, z')

        :param threshold: Sparsity threshold for SINDy
        :type threshold: float
        :param differentiation_method: Diff. method for SINDy.
        :type differentiation_method: str
        :param plot_result: Flag for plotting discovered model
        :type plot_result: bool
        :param save_path: Path to save the plot (without extension). If provided,
            saves both .pdf and .png versions. Example: "results/latent_comparison"
        :type save_path: str or None
        """

        # TODO: allow users to pass any differentiation method
        #   and implement MIOSR option
        if differentiation_method == "finite":
            self._differentiation_method = ps.differentiation.FiniteDifference()
        elif differentiation_method == "smoothed finite":
            self._differentiation_method = ps.differentiation.SmoothedFiniteDifference()

        if self._train_data is None:
            raise ValueError("You need to call `fit` prior to recovering SINDy states.")

        # Normalized SINDy-SHRED latent space trajectories for post-hoc model discovery
        gru_outs = self.gru_normalize(data_type="train")
        z = gru_outs.detach().cpu().numpy()
        self._gru_outs = z

        if self._ode_order == 1:
            # 1st order ODE: z' = f(z)
            model = ps.SINDy(
                optimizer=ps.STLSQ(threshold=threshold, alpha=0.05),
                differentiation_method=self._differentiation_method,
                feature_library=ps.PolynomialLibrary(degree=self._poly_order),
            )
            model.fit(z, t=self._dt)
            self._model = model

            if self._verbose:
                print("SINDy-derived dynamical equation (1st order):\n")
                model.print()

            # Plot the discovered SINDy model
            if plot_result or save_path:
                self.sindy_simulate(z)
                x_sim = self._x_sim
                t_train = np.arange(0, len(z) * self._dt, self._dt)
                fig, ax = plt.subplots(self._latent_dim, sharex=True, sharey=True)
                if self._latent_dim == 1:
                    ax = [ax]
                for i in range(self._latent_dim):
                    ax[i].plot(
                        t_train,
                        gru_outs[:, i].detach().cpu().numpy(),
                        label="SINDy-SHRED",
                    )
                    ax[i].plot(t_train, x_sim[:, i], "k--", label="identified model")
                    ax[i].set_ylabel(rf"$z_{{{i}}}$ (-)")
                    if i == self._latent_dim - 1:
                        ax[i].set_xlabel("time (n steps)")
                        ax[i].legend()
                if save_path:
                    fig.savefig(f"{save_path}.pdf", bbox_inches="tight", dpi=300)
                    fig.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
                if plot_result:
                    plt.show()
                else:
                    plt.close(fig)

        elif self._ode_order == 2:
            # 2nd order ODE: z'' = f(z, z')
            # Convert to state-space form: [z, v]' = [v, f(z, v)] where v = z'

            # Estimate velocity using finite difference
            dz, z_trimmed = self.estimate_velocity(z, self._dt)
            self._z_trimmed = z_trimmed
            self._dz = dz

            # Store initial conditions for prediction
            self._init_z = z_trimmed[0, :]
            self._init_dz = dz[0, :]

            # Concatenate position and velocity for state-space representation
            # State x = [z, v] where v = dz/dt
            x_state = np.concatenate([z_trimmed, dz], axis=1)

            # Compute acceleration for fitting: ddz = d(dz)/dt
            ddz = (dz[1:] - dz[:-1]) / self._dt

            # Align arrays: x_state needs to be trimmed to match ddz length
            x_state_trimmed = x_state[:-1]

            # Build derivative array: dx/dt = [dz, ddz]
            dz_trimmed = dz[:-1]
            x_dot = np.concatenate([dz_trimmed, ddz], axis=1)

            # Fit SINDy model with explicit derivatives
            model = ps.SINDy(
                optimizer=ps.STLSQ(threshold=threshold, alpha=0.05),
                feature_library=ps.PolynomialLibrary(degree=self._poly_order),
            )
            model.fit(x_state_trimmed, t=self._dt, x_dot=x_dot)
            self._model = model

            if self._verbose:
                print(
                    "SINDy-derived dynamical equation (2nd order, state-space form):\n"
                )
                print("State: [z_0, ..., z_{n-1}, v_0, ..., v_{n-1}]")
                print("where v = dz/dt\n")
                model.print()

            # Plot the discovered SINDy model
            if plot_result or save_path:
                self.sindy_simulate(z)
                x_sim = self._x_sim
                t_train = np.arange(0, len(z_trimmed) * self._dt, self._dt)
                fig, ax = plt.subplots(
                    self._latent_dim * 2, sharex=True, figsize=(8, 2 * self._latent_dim)
                )
                if self._latent_dim * 2 == 1:
                    ax = [ax]
                # Plot positions
                for i in range(self._latent_dim):
                    ax[i].plot(t_train, z_trimmed[:, i], label="SINDy-SHRED")
                    ax[i].plot(t_train, x_sim[:, i], "k--", label="identified model")
                    ax[i].set_ylabel(rf"$z_{{{i}}}$ (-)")
                    if i == 0:
                        ax[i].legend()
                # Plot velocities
                for i in range(self._latent_dim):
                    ax[self._latent_dim + i].plot(t_train, dz[:, i], label="estimated")
                    ax[self._latent_dim + i].plot(
                        t_train,
                        x_sim[:, self._latent_dim + i],
                        "k--",
                        label="identified model",
                    )
                    ax[self._latent_dim + i].set_ylabel(rf"$\dot{{z}}_{{{i}}}$ (-)")
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
            raise ValueError(f"ode_order must be 1 or 2, got {self._ode_order}")

    def sindy_simulate(self, x):
        """Integrate the SINDy model forward in time.

        Assumes the period for integration is a time series of len(x) with time steps
        given by `dt`. The initial value for the latent space is given as the first
        latent space value from the SINDy-SHRED model.

        For 2nd order systems, integrates in state-space form [z, dz].

        :param x: SINDy-SHRED latent space (e.g., from `SINDy_SHRED_net.gru_outputs`)
        :type x: numpy.ndarray
        """
        model = self._model

        if self._ode_order == 1:
            t_train = np.arange(0, len(x) * self._dt, self._dt)
            init_cond = np.zeros(self._latent_dim)
            init_cond[: self._latent_dim] = x[0, :]
            self._x_sim = model.simulate(init_cond, t_train)

        elif self._ode_order == 2:
            # For 2nd order, use trimmed data length and state-space initial conditions
            z_trimmed = self._z_trimmed
            t_train = np.arange(0, len(z_trimmed) * self._dt, self._dt)

            # Initial condition is [z(0), dz(0)]
            init_cond = np.concatenate([self._init_z, self._init_dz])
            self._x_sim = model.simulate(init_cond, t_train)

    def auto_tune_threshold(
        self,
        thresholds=None,
        metric="sparsity_stable",
        test_steps=None,
        divergence_threshold=1e6,
        verbose=None,
        adaptive=True,
        scale_factor=0.3,
        n_thresholds=10,
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
            verbose = self._verbose

        if self._gru_outs is None:
            # Need to get normalized latent space first
            gru_outs = self.gru_normalize(data_type="train")
            x_train = gru_outs.detach().cpu().numpy()
        else:
            x_train = self._gru_outs

        if adaptive:
            # Nonparametric approach: determine thresholds from least-squares solution
            if verbose:
                print(
                    "Computing least-squares solution to determine threshold range..."
                )

            # Fit with threshold=0 to get the full least-squares solution
            ls_model = ps.SINDy(
                optimizer=ps.STLSQ(threshold=0.0, alpha=0.05),
                differentiation_method=self._differentiation_method,
                feature_library=ps.PolynomialLibrary(degree=self._poly_order),
            )
            ls_model.fit(x_train, t=self._dt)

            # Get max absolute coefficient value
            coeffs = ls_model.coefficients()
            max_coeff = np.max(np.abs(coeffs))
            max_threshold = scale_factor * max_coeff

            # Generate evenly spaced thresholds from 0 to max_threshold
            thresholds = np.linspace(0, max_threshold, n_thresholds)

            if verbose:
                print(f"Max |coefficient|: {max_coeff:.4f}")
                print(
                    f"Max threshold (scale_factor={scale_factor}): {max_threshold:.4f}"
                )
                print(f"Testing {n_thresholds} thresholds: {thresholds}")
        elif thresholds is None:
            thresholds = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])

        if test_steps is None:
            test_steps = self._test_length if self._test_length else 100

        # Get test latent space for validation
        gru_test = self.gru_normalize(data_type="test")
        x_test = gru_test.detach().cpu().numpy()

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

            # Fit SINDy with this threshold
            model = ps.SINDy(
                optimizer=ps.STLSQ(threshold=thresh, alpha=0.05),
                differentiation_method=self._differentiation_method,
                feature_library=ps.PolynomialLibrary(degree=self._poly_order),
            )
            model.fit(x_train, t=self._dt)

            # Count nonzero coefficients (sparsity)
            n_nonzero = np.count_nonzero(model.coefficients())
            results["sparsity"].append(n_nonzero)

            # Test stability via forward integration
            try:
                t_test = np.arange(0, test_steps * self._dt, self._dt)
                init_cond = x_test[0, :]
                x_sim = model.simulate(init_cond, t_test)

                # Check for divergence
                if np.any(np.abs(x_sim) > divergence_threshold) or np.any(
                    np.isnan(x_sim)
                ):
                    is_stable = False
                    mse = np.inf
                else:
                    is_stable = True
                    # Compute MSE against test data
                    n_compare = min(len(x_sim), len(x_test))
                    mse = np.mean((x_sim[:n_compare] - x_test[:n_compare]) ** 2)
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

        # Re-fit with best threshold and store
        self.sindy_identify(threshold=best_threshold, plot_result=False)

        return best_threshold, results

    def sindy_predict(
        self, t=None, init_cond=None, init_from="train", return_velocity=False
    ):
        """Predict the latent space using the discovered SINDy model.

        This function integrates the discovered SINDy model forward in time starting
        from a specified initial condition.

        For 1st order systems: predicts z(t) given z(0)
        For 2nd order systems: predicts [z(t), dz(t)] given [z(0), dz(0)]

        :param t: Time array for integration. If None, uses test length with dt.
        :type t: numpy.ndarray or None
        :param init_cond: Initial condition for integration. If None, determined by
            init_from parameter. For 2nd order, should be [z0, dz0].
        :type init_cond: numpy.ndarray or None
        :param init_from: Where to get initial condition if init_cond is None.
            "test" - use first test point's latent state (default, for comparing
                     predictions against test ground truth)
            "train" - use last training point's latent state (for forecasting
                      beyond training data)
        :type init_from: str
        :param return_velocity: For 2nd order systems, whether to return [z, dz] or
            just z. Default is False (return only position).
        :type return_velocity: bool
        :return: Predicted latent space trajectories.
            For 1st order, shape (n_times, latent_dim).
            For 2nd order with return_velocity=False, shape (n_times, latent_dim).
            For 2nd order with return_velocity=True, shape (n_times, 2*latent_dim).
        :rtype: numpy.ndarray
        """

        if t is None:
            dt = self._dt
            t = np.arange(0, self._test_length * dt, dt)

        model = self._model

        if self._ode_order == 1:
            if init_cond is None:
                init_cond = np.zeros(self._latent_dim)
                if init_from == "test":
                    gru_test_np = (
                        self.gru_normalize(data_type="test").detach().cpu().numpy()
                    )
                    init_cond[: self._latent_dim] = gru_test_np[0, :]
                elif init_from == "train":
                    gru_train_np = (
                        self.gru_normalize(data_type="train").detach().cpu().numpy()
                    )
                    init_cond[: self._latent_dim] = gru_train_np[-1, :]
                else:
                    raise ValueError(
                        f"init_from must be 'test' or 'train', got '{init_from}'"
                    )

            x_predict = model.simulate(init_cond, t)
            return x_predict

        elif self._ode_order == 2:
            # For 2nd order, initial condition is [z0, dz0]
            if init_cond is None:
                if init_from == "test":
                    # Get test data and estimate velocity
                    gru_test_np = (
                        self.gru_normalize(data_type="test").detach().cpu().numpy()
                    )
                    dz_test, z_test = self.estimate_velocity(gru_test_np, self._dt)
                    init_z = z_test[0, :]
                    init_dz = dz_test[0, :]
                elif init_from == "train":
                    # Use stored training data velocities
                    init_z = self._z_trimmed[-1, :]
                    init_dz = self._dz[-1, :]
                else:
                    raise ValueError(
                        f"init_from must be 'test' or 'train', got '{init_from}'"
                    )
                init_cond = np.concatenate([init_z, init_dz])

            # Integrate in state-space form
            x_predict = model.simulate(init_cond, t)

            if return_velocity:
                return x_predict
            else:
                # Return only position (first latent_dim columns)
                return x_predict[:, : self._latent_dim]
        else:
            raise ValueError(f"ode_order must be 1 or 2, got {self._ode_order}")

    def gru_normalize(self, data_type=None):
        """Get grus and normalize them by the training data.

        :param data_type: Specifies which data split to return. All data splits are
        normalized by the training data latent space to be on the scale [0, 1]. Must
        be one of `train`, `validate`, or `test`.
        :type data_type: str
        :return gru_outs: Normalized latent space variables.
        :rtype gru_outs: torch.tensor
        """

        # gru_out_train, _ = self._shred.gru_outputs(self._train_data.X, sindy=True)
        # # What is this indexing doing?
        # gru_out_train = gru_out_train[:, 0, :]

        gru_out_train = self.get_gru(data_type="train")
        gru_outs = self.get_gru(data_type=data_type)

        # Normalization
        for n in range(self._latent_dim):
            gru_outs[:, n] = (gru_outs[:, n] - torch.min(gru_out_train[:, n])) / (
                torch.max(gru_out_train[:, n]) - torch.min(gru_out_train[:, n])
            )
        gru_outs = 2 * gru_outs - 1
        return gru_outs

    def get_gru(self, data_type=None):
        if data_type is None:
            data_type = "train"

        if data_type == "train":
            gru_outs, _ = self._shred.gru_outputs(self._train_data.X, sindy=True)
        elif data_type == "validate":
            gru_outs, _ = self._shred.gru_outputs(self._valid_data.X, sindy=True)
        elif data_type == "test":
            gru_outs, _ = self._shred.gru_outputs(self._test_data.X, sindy=True)
        else:
            raise ValueError("Unrecognized `data_type` provided.")
        gru_outs = gru_outs[:, 0, :]

        return gru_outs

    def shred_decode(self, z):
        """Convert SINDy simulated latent space into scaled physical space using SHRED.

        A SINDy-SHRED model needs to have been fit first.

        Note: For 2nd order systems, this method expects only position z, not the
        full state [z, dz]. Use sindy_predict() with return_velocity=False (default)
        to get the correct input format.

        :param z: Latent space trajectories (i.e. from SINDy model), shape (n_samples, latent_dim)
        :type z: numpy.ndarray or torch.tensor
        :return: Physical, high-dimensional signal in min-max scaled space.
        :rtype: numpy.ndarray
        """
        # Ensure latent space trajectories are a numpy array if needed
        z = np.array(z)

        # Convert scaling from [-1, 1] to [0, 1]
        z = (z + 1) / 2

        # Perform the Min-Max reverse transformation using the training SINDy-SHRED
        # latent space
        gru_out_train, _ = self._shred.gru_outputs(self._train_data.X, sindy=True)
        gru_out_train = gru_out_train[:, 0, :]
        gru_out_train = gru_out_train.detach().cpu().numpy()
        # Each latent space dimension is normalized by itself.
        for n in range(self._latent_dim):
            z[:, n] = z[:, n] * (
                np.max(gru_out_train[:, n]) - np.min(gru_out_train[:, n])
            ) + np.min(gru_out_train[:, n])

        # Perform the decoder reconstruction using the transformed SINDy-simulated data
        latent_pred_sindy = torch.FloatTensor(z).to(
            self._device
        )  # Convert to torch tensor for reconstruction

        # Pass the SINDy-simulated latent space data through the decoder
        decoder_model = self._shred
        output_sindy = decoder_model.linear1(latent_pred_sindy)
        output_sindy = decoder_model.dropout(output_sindy)
        output_sindy = torch.nn.functional.relu(output_sindy)
        output_sindy = decoder_model.linear2(output_sindy)
        output_sindy = decoder_model.dropout(output_sindy)
        output_sindy = torch.nn.functional.relu(output_sindy)
        output_sindy = decoder_model.linear3(output_sindy)

        # Detach and convert the reconstructed data back to numpy for visualization
        output_sindy_np = output_sindy.detach().cpu().numpy()

        return output_sindy_np

    def sensor_recon(self, data_type="test", return_scaled=False):
        """Reconstruct full state from sparse sensor measurements.

        This method uses the trained SHRED network to reconstruct the full
        high-dimensional state from the sparse sensor trajectory inputs.

        Parameters
        ----------
        data_type : str, optional
            Which data split to reconstruct. One of "train", "validate", or "test".
            Default is "test".
        return_scaled : bool, optional
            If True, return scaled (0-1) values. If False, return in original scale.
            Default is False.

        Returns
        -------
        reconstructions : numpy.ndarray
            Reconstructed high-dimensional states, shape (n_samples, state_dim).
        """
        if data_type == "train":
            data = self._train_data
        elif data_type == "validate":
            data = self._valid_data
        elif data_type == "test":
            data = self._test_data
        else:
            raise ValueError(
                f"data_type must be 'train', 'validate', or 'test', got '{data_type}'"
            )

        # Get reconstructions from SHRED
        recons_scaled = self._shred(data.X).detach().cpu().numpy()

        if return_scaled:
            return recons_scaled
        else:
            # Inverse transform to original scale
            return self._scaler.inverse_transform(recons_scaled)

    def forecast(self, n_steps=None, init_from="train", return_scaled=False):
        """Forecast future states using the discovered SINDy model.

        This is a convenience method that combines sindy_predict() and shred_decode()
        to produce forecasts in physical space.

        Parameters
        ----------
        n_steps : int, optional
            Number of time steps to forecast. If None, uses test length.
        init_from : str, optional
            Where to get initial condition. "test" uses first test point,
            "train" uses last training point. Default is "train".
        return_scaled : bool, optional
            If True, return scaled (0-1) values. If False, return in original scale.
            Default is False.

        Returns
        -------
        forecast : numpy.ndarray
            Forecasted high-dimensional states, shape (n_steps, state_dim).
        """
        if self._model is None:
            raise ValueError("Must call sindy_identify() before forecasting.")

        if n_steps is None:
            n_steps = self._test_length

        # Predict latent trajectories
        t = np.arange(0, n_steps * self._dt, self._dt)
        z_predict = self.sindy_predict(t=t, init_from=init_from)

        # Decode to physical space
        forecast_scaled = self.shred_decode(z_predict)

        if return_scaled:
            return forecast_scaled
        else:
            return self._scaler.inverse_transform(forecast_scaled)

    # Aliases for clearer API
    predict_latent = sindy_predict
    decode_to_physical = shred_decode
