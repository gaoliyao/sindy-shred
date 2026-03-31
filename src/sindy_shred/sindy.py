# Author: Mars Gao
# Date: Nov/17/2021

"""SINDy library construction functions.

This module provides functions to build SINDy libraries for sparse dynamics
identification, supporting both 1st and 2nd order ODEs.
"""

import torch
from scipy.special import binom


def build_sindy_library(z, poly_order, include_sine=False, dz=None):
    """Build SINDy library for 1st or 2nd order ODEs.

    This is the unified interface for building SINDy libraries. For 1st order
    systems (dz=None), builds library Θ(z). For 2nd order systems (dz provided),
    builds library Θ(z, dz) where dz represents velocity.

    Parameters
    ----------
    z : torch.Tensor
        Position variables, shape (n_samples, latent_dim).
    poly_order : int
        Polynomial order for the library (max 5).
    include_sine : bool, optional
        Whether to include sine terms. Default is False.
    dz : torch.Tensor, optional
        Velocity variables for 2nd order systems, shape (n_samples, latent_dim).
        If provided, builds 2nd order library. Default is None.

    Returns
    -------
    library : torch.Tensor
        SINDy library, shape (n_samples, library_dim).
    """
    device = z.device

    if dz is not None:
        # 2nd order: combine position and velocity
        z_combined = torch.cat([z, dz], dim=1)
        n_vars = z_combined.shape[1]
    else:
        # 1st order: just position
        z_combined = z
        n_vars = z.shape[1]

    library = [torch.ones(z.shape[0], device=device)]

    # Linear terms
    for i in range(n_vars):
        library.append(z_combined[:, i])

    # Quadratic terms
    if poly_order > 1:
        for i in range(n_vars):
            for j in range(i, n_vars):
                library.append(z_combined[:, i] * z_combined[:, j])

    # Cubic terms
    if poly_order > 2:
        for i in range(n_vars):
            for j in range(i, n_vars):
                for k in range(j, n_vars):
                    library.append(z_combined[:, i] * z_combined[:, j] * z_combined[:, k])

    # Quartic terms
    if poly_order > 3:
        for i in range(n_vars):
            for j in range(i, n_vars):
                for k in range(j, n_vars):
                    for p in range(k, n_vars):
                        library.append(
                            z_combined[:, i] * z_combined[:, j] * z_combined[:, k] * z_combined[:, p]
                        )

    # Quintic terms
    if poly_order > 4:
        for i in range(n_vars):
            for j in range(i, n_vars):
                for k in range(j, n_vars):
                    for p in range(k, n_vars):
                        for q in range(p, n_vars):
                            library.append(
                                z_combined[:, i] * z_combined[:, j] * z_combined[:, k]
                                * z_combined[:, p] * z_combined[:, q]
                            )

    # Sine terms
    if include_sine:
        for i in range(n_vars):
            library.append(torch.sin(z_combined[:, i]))

    return torch.stack(library, dim=1)


def sindy_library_torch(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library.
    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape
        is number of time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value
        is 5.
        include_sine - Boolean, whether or not to include sine terms in the library.
        Default False.
    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time
        points by number of library functions. The number of library functions is
        determined by the number of state variables of the input, the polynomial
        order, and whether sines are included.
    """
    device = z.device
    library = [torch.ones(z.shape[0], device=device)]

    for i in range(latent_dim):
        library.append(z[:, i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(torch.multiply(z[:, i], z[:, j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(z[:, i] * z[:, j] * z[:, k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for q in range(p, latent_dim):
                            library.append(
                                z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q]
                            )

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:, i]))

    return torch.stack(library, axis=1)


def e_sindy_library_torch(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library.
    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.
    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    device = z.device
    library = [torch.ones(z.shape[0]).to(device)]

    for i in range(latent_dim):
        library.append(z[:, i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(torch.multiply(z[:, i], z[:, j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(z[:, i] * z[:, j] * z[:, k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for q in range(p, latent_dim):
                            library.append(
                                z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q]
                            )

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:, i]))

    return torch.stack(library, axis=1)


def sindy_library_torch_order2(z, dz, latent_dim, poly_order, include_sine=False):
    """Build SINDy library for 2nd order systems.

    Wrapper around build_sindy_library for backward compatibility.

    Parameters
    ----------
    z : torch.Tensor
        Position variables, shape (n_samples, latent_dim).
    dz : torch.Tensor
        Velocity variables, shape (n_samples, latent_dim).
    latent_dim : int
        Number of latent dimensions (unused, inferred from z).
    poly_order : int
        Polynomial order for the library.
    include_sine : bool, optional
        Whether to include sine terms. Default is False.

    Returns
    -------
    library : torch.Tensor
        SINDy library, shape (n_samples, library_dim).
    """
    return build_sindy_library(z, poly_order, include_sine=include_sine, dz=dz)


def e_sindy_library_torch_order2(z, dz, latent_dim, poly_order, include_sine=False):
    """Build ensemble SINDy library for 2nd order systems.

    Wrapper around build_sindy_library for backward compatibility.
    Same as sindy_library_torch_order2.

    Parameters
    ----------
    z : torch.Tensor
        Position variables, shape (n_samples, latent_dim).
    dz : torch.Tensor
        Velocity variables, shape (n_samples, latent_dim).
    latent_dim : int
        Number of latent dimensions (unused, inferred from z).
    poly_order : int
        Polynomial order for the library.
    include_sine : bool, optional
        Whether to include sine terms. Default is False.

    Returns
    -------
    library : torch.Tensor
        SINDy library, shape (n_samples, library_dim).
    """
    return build_sindy_library(z, poly_order, include_sine=include_sine, dz=dz)


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l
