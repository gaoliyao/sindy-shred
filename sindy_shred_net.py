"""SINDy-SHRED: Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder.
"""SINDy-SHRED: Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder.

This module provides the core neural network components for the SINDy-SHRED architecture,
which combines recurrent neural networks with sparse dynamics identification for
interpretable spatio-temporal modeling.

Classes
-------
SINDy_SHRED_net
    Main SINDy-SHRED neural network architecture.
E_SINDy
    Ensemble SINDy module for sparse dynamics learning.

Functions
---------
fit
    Training function for SINDy-SHRED models.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from sindy import sindy_library_torch, e_sindy_library_torch
from utils import get_device

device = get_device()


class E_SINDy(torch.nn.Module):
    """Ensemble SINDy module for learning sparse dynamics in latent space.

    This module learns sparse coefficient matrices that govern the dynamics
    of the latent space variables using an ensemble approach.

    Parameters
    ----------
    num_replicates : int
        Number of ensemble members for coefficient estimation.
    latent_dim : int
        Dimension of the latent space.
    library_dim : int
        Number of candidate functions in the SINDy library.
    poly_order : int
        Polynomial order for the library.
    include_sine : bool
        Whether sine terms are included in the library.
    device : str or torch.device, optional
        Device to use. Default is None (auto-detect).
    """

    def __init__(
        self,
        num_replicates,
        latent_dim,
        library_dim,
        poly_order,
        include_sine,
        device=None,
    ):
        super().__init__()
        self.num_replicates = num_replicates
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.device = device
        self.coefficients = torch.ones(
            num_replicates, library_dim, latent_dim, requires_grad=True
        )
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        self.coefficient_mask = torch.ones(
            num_replicates, library_dim, latent_dim, requires_grad=False
        ).to(self.device)
        self.coefficients = torch.nn.Parameter(self.coefficients)

        if device is None:
            self.device = get_device()
        else:
            self.device = device

    def forward(self, h_replicates, dt):
        num_data, num_replicates, latent_dim = h_replicates.shape
        h_replicates = h_replicates.reshape(num_data * num_replicates, latent_dim)
        library_Thetas = e_sindy_library_torch(
            h_replicates, self.latent_dim, self.poly_order, self.include_sine
        )
        library_Thetas = library_Thetas.reshape(
            num_data, num_replicates, self.library_dim
        )
        h_replicates = h_replicates.reshape(num_data, num_replicates, latent_dim)
        h_replicates = (
            h_replicates
            + torch.einsum(
                "ijk,jkl->ijl",
                library_Thetas,
                (self.coefficients * self.coefficient_mask),
            )
            * dt
        )
        return h_replicates

    def thresholding(self, threshold, base_threshold=0):
        threshold_tensor = torch.full_like(self.coefficients, threshold)
        for i in range(self.num_replicates):
            threshold_tensor[i] = (
                threshold_tensor[i] * 10 ** (0.2 * i - 1) + base_threshold
            )
        self.coefficient_mask = torch.abs(self.coefficients) > threshold_tensor
        self.coefficients.data = self.coefficient_mask * self.coefficients.data


class SINDy_SHRED_net(torch.nn.Module):
    """SINDy-SHRED neural network architecture.

    Combines a GRU encoder with a shallow decoder network (SDN) and integrates
    Sparse Identification of Nonlinear Dynamics (SINDy) for learning interpretable
    latent space dynamics.

    Parameters
    ----------
    input_size : int
        Number of sensors (input features per time step).
    output_size : int
        Dimension of the high-dimensional state to reconstruct.
    hidden_size : int, optional
        Size of the GRU hidden state (latent dimension). Default is 64.
    hidden_layers : int, optional
        Number of GRU layers. Default is 1.
    l1 : int, optional
        Size of the first decoder layer. Default is 350.
    l2 : int, optional
        Size of the second decoder layer. Default is 400.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    library_dim : int, optional
        Dimension of the SINDy library. Default is 10.
    poly_order : int, optional
        Polynomial order for SINDy library. Default is 3.
    include_sine : bool, optional
        Whether to include sine terms in SINDy library. Default is False.
    dt : float, optional
        Time step for SINDy integration. Default is 0.03.
    device : str or torch.device, optional
        Device to use. Default is None (auto-detect).
    multi_step : int, optional
        Number of multi-step predictions for training. Default is None.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        hidden_layers=1,
        l1=350,
        l2=400,
        dropout=0.0,
        library_dim=10,
        poly_order=3,
        include_sine=False,
        dt=0.03,
        device=None,
        multi_step=None,
    ):
        if device is None:
            device = get_device()
        self.device = device

        super(SINDy_SHRED_net, self).__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
        ).to(self.device)
        self.num_replicates = 5
        self.num_euler_steps = 3
        self.e_sindy = E_SINDy(
            self.num_replicates,
            hidden_size,
            library_dim,
            poly_order,
            include_sine,
            device=self.device,
        )

        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.dt = dt
        self.multi_step = multi_step
        self.max_multi_step = 10

    def forward(self, x, sindy=False):
        h_0 = torch.zeros(
            (self.hidden_layers, x.size(0), self.hidden_size),
            dtype=torch.float,
            device=self.device,
        )
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()

        _, h_out = self.gru(x, h_0)
        h_out = h_out[-1].view(-1, self.hidden_size)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear3(output)
        with torch.autograd.set_detect_anomaly(True):
            if sindy:
                steps = int(self.multi_step) if self.multi_step is not None else 1
                steps = max(1, steps)
                if steps > self.max_multi_step:
                    raise ValueError(
                        f"multi_step={steps} exceeds max_multi_step="
                        f"{self.max_multi_step}"
                    )
                start_latents = h_out[:-steps, :]
                target_latents = h_out[steps:, :]
                ht_replicates = start_latents.unsqueeze(1).repeat(
                    1, self.num_replicates, 1
                )
                total_substeps = self.num_euler_steps * steps
                for _ in range(total_substeps):
                    ht_replicates = self.e_sindy(
                        ht_replicates, dt=self.dt / float(self.num_euler_steps)
                    )
                h_out_replicates = target_latents.unsqueeze(1).repeat(
                    1, self.num_replicates, 1
                )
                return output, h_out_replicates, ht_replicates
        return output

    def gru_outputs(self, x, sindy=False):
        h_0 = torch.zeros(
            (self.hidden_layers, x.size(0), self.hidden_size),
            dtype=torch.float,
            device=self.device,
        )
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
        _, h_out = self.gru(x, h_0)
        h_out = h_out[-1].view(-1, self.hidden_size)

        if sindy:
            # Always return 1-step pairs for external consumers to preserve expected
            # length
            if h_out.shape[0] <= 1:
                return None, None
            start_latents = h_out[:-1, :]
            target_latents = h_out[1:, :]
            ht_replicates = start_latents.unsqueeze(1).repeat(1, self.num_replicates, 1)
            for _ in range(self.num_euler_steps):
                ht_replicates = self.e_sindy(
                    ht_replicates, dt=self.dt / float(self.num_euler_steps)
                )
            h_out_replicates = target_latents.unsqueeze(1).repeat(
                1, self.num_replicates, 1
            )
            return h_out_replicates, ht_replicates
        return None

    def sindys_threshold(self, threshold):
        self.e_sindy.thresholding(threshold)

    def decode(self, h):
        """Decode latent vectors to physical space.

        Parameters
        ----------
        h : torch.Tensor
            Latent vectors of shape (batch_size, hidden_size).

        Returns
        -------
        torch.Tensor
            Decoded output of shape (batch_size, output_size).
        """
        output = self.linear1(h)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear3(output)
        return output


def fit(
    model,
    train_dataset,
    valid_dataset,
    batch_size=64,
    num_epochs=4000,
    lr=1e-3,
    sindy_regularization=1.0,
    optimizer="AdamW",
    verbose=False,
    threshold=0.5,
    base_threshold=0.0,
    patience=20,
    thres_epoch=100,
    weight_decay=0.01,
    multi_step=1,
    grad_clip_norm=1000,
):
    criterion = torch.nn.MSELoss()
    if optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    # allow overriding model's multi_step per training run (default keeps old behavior)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size + 1)
    for epoch in range(1, num_epochs + 1):
        current_bs = int(np.random.randint(batch_size // 2, batch_size + 1))
        for idx_batch, data in enumerate(train_loader):
            data = data[:current_bs]
            model.train()
            outputs, h_gru, h_sindy = model(data[0], sindy=True)
            if outputs is None:
                print(f"skip this iter because outputs is None")
                continue
            optimizer.zero_grad()
            loss = (
                criterion(outputs, data[1])
                + criterion(h_gru, h_sindy) * sindy_regularization
                + torch.abs(torch.mean(h_gru)) * 0.1
            )
            if not torch.isfinite(loss).all():
                print("Non-finite loss encountered; skipping optimization step")
                model.zero_grad(set_to_none=True)
                continue

            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        print(epoch, ":", loss)

        if epoch % thres_epoch == 0 and epoch != 0:
            model.e_sindy.thresholding(
                threshold=threshold, base_threshold=base_threshold
            )
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(val_outputs - valid_dataset.Y)
                val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)
            if verbose:
                print("Training epoch " + str(epoch))
                print("Error " + str(val_error_list[-1]))
            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter == patience:
                return torch.tensor(val_error_list).cpu()
    return torch.tensor(val_error_list).detach().cpu().numpy()
