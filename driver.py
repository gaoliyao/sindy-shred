import torch
import sindy
import pysindy as ps
import sindy_shred
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter("ignore", UserWarning)


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Takes input sequence of sensor measurements with shape (batch size, lags,
    num_sensors) and corresponding measurements of high-dimensional state,
    return Torch dataset"""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


class sindy_shred_driver:
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
        if device is None:
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = device

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
        # if sample_size is None:
        #     sample_size = int(self._n_time_dim * 0.6)
        # else:
        #     sample_size = self._train_length

        if mode == "forecast":
            if test_length is None:
                test_length = self._n_time_dim - (train_length + validate_length)

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

        shred = sindy_shred.SINDy_SHRED(
            self._num_sensors,
            self._n_space_dim,
            dt=self._dt,
            **self._sindy_shred_kwargs,
        ).to(self._device)

        validation_errors = sindy_shred.fit(
            shred, self._train_data, self._valid_data, **self._fit_kwargs
        )

        self._shred = shred

    def sindy_identify(
        self, threshold, differentiation_method="finite", plot_result=True
    ):
        """Post-hoc model discovery with SINDy using SHRED latent space trajectories.

        :param threshold: Sparsity threshold for SINDy
        :type threshold: float
        :param differentiation_method: Diff. method for SINDy.
        :type differentiation_method: str
        :param plot_result: Flag for plotting discovered model
        :type plot_result: bool
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
        x = gru_outs.detach().cpu().numpy()
        self._gru_outs = x

        # SINDy discovery
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=threshold, alpha=0.05),
            differentiation_method=self._differentiation_method,
            feature_library=ps.PolynomialLibrary(degree=self._poly_order),
        )

        model.fit(x, t=self._dt, ensemble=True)
        self._model = model

        if self._verbose:
            print("SINDy-derived dynamical equation:\n")
            model.print()

        # Plot the discovered SINDy model
        if plot_result:
            self.sindy_simulate(x)
            x_sim = self._x_sim
            t_train = np.arange(0, len(x) * self._dt, self._dt)
            fig, ax = plt.subplots(self._latent_dim, sharex=True, sharey=True)
            for i in range(self._latent_dim):
                ax[i].plot(
                    t_train, gru_outs[:, i].detach().cpu().numpy(), label="SINDy-SHRED"
                )
                ax[i].plot(t_train, x_sim[:, i], "k--", label="identified model")
                ax[i].set_ylabel(rf"$z_{{{i}}}$ (-)")
                if i == self._latent_dim - 1:
                    ax[i].set_xlabel("time (n steps)")
                    ax[i].legend()
            plt.show()

    # @ToDo: Integrate this function with sindy_predict, below.
    def sindy_simulate(self, x):
        """Integrate the SINDy model forward in time.

        Assumes the period for integration is a time series of len(x) with time steps
        given by `dt`. The initial value for the latent space is given as the first
        latent space value from the SINDy-SHRED model.

        :param x: SINDy-SHRED latent space (e.g., from `SINDy_SHRED.gru_outputs`)
        :type x: numpy.ndarray
        """
        model = self._model
        t_train = np.arange(0, len(x) * self._dt, self._dt)
        init_cond = np.zeros(self._latent_dim)
        init_cond[: self._latent_dim] = x[0, :]
        self._x_sim = model.simulate(init_cond, t_train)

    def sindy_predict(self, t=None, init_cond=None):
        """Predict the latent space using the discovered SINDy model.

        This function is distinct from sindy_simulate because it assumes the initial
        condition is the last latent space value from the training data and
        integrates forward through a time series of `t`.

        """

        if t is None:
            # Predicts from the last training data point over the length of the test
            # data.
            dt = self._dt
            t = np.arange(0, self._test_length * dt, dt)

        model = self._model

        if init_cond is None:
            # The intial condition is the last training latent space values.
            init_cond = np.zeros(self._latent_dim)
            gru_train_np = self.gru_normalize(data_type="train").detach().cpu().numpy()
            init_cond[: self._latent_dim] = gru_train_np[-1, :]

        x_predict = model.simulate(init_cond, t)

        return x_predict

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
        """Convert SINDy simulated latent space into scaled physical space using SHRED

        A SINDy-SHRED model needs to have been fit first.

        :param z: Latent space trajectories (i.e. from SINDy model)
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
