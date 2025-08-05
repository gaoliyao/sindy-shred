# Author: Mars Gao
# Date: Nov/17/2021

# Include necessary packages
import torch
from scipy.special import binom

def sindy_library_torch(z, latent_dim, poly_order, include_sine=False):
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
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    library = [torch.ones(z.shape[0]).to(device)]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(torch.multiply(z[:,i], z[:,j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

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
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(torch.multiply(z[:,i], z[:,j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, axis=1)

def sindy_library_torch_version2(z, latent_dim, poly_order=3, include_sine=False):
    """
    Efficiently construct the SINDy library tensor including up to the specified polynomial order,
    focusing on reducing computational overhead for cubic terms.
    Assumes `z` is already on the appropriate device (GPU if available).
    """
    device = z.device
    # Start with the constant term (bias term)
    library = [torch.ones(z.shape[0], 1, device=device)]
    
    # Linear terms
    library.append(z)
    
    if poly_order >= 2:
        # Efficient quadratic term generation using broadcasting
        z_expanded = z.unsqueeze(-1)
        quadratic_terms = z_expanded * z_expanded.transpose(1, 2)
        idx_upper_tri = torch.triu_indices(z.shape[1], z.shape[1], device=device)
        quadratic_terms = quadratic_terms[:, idx_upper_tri[0], idx_upper_tri[1]]
        library.append(quadratic_terms)

    if poly_order >= 3:
        # Efficient cubic term generation using a single loop
        cubic_terms = []
        for i in range(z.shape[1]):
            for j in range(i, z.shape[1]):
                for k in range(j, z.shape[1]):
                    cubic_term = z[:, i] * z[:, j] * z[:, k]
                    cubic_terms.append(cubic_term.unsqueeze(1))
        cubic_terms = torch.cat(cubic_terms, dim=1)
        library.append(cubic_terms)

    # Concatenate all terms to form the library
    library = torch.cat(library, dim=1)
    
    if include_sine:
        sine_terms = torch.sin(z)
        library = torch.cat([library, sine_terms], dim=1)

    return library

def sindy_library_torch_order2(z, dz, latent_dim, poly_order, include_sine=False, sine_k = 1.0, print_names = "False"):
    """
    Build the SINDy library for a second order system. This is essentially the same as for a first
    order system, but library terms are also built for the derivatives.
    """
    library = [torch.ones(z.shape[0]).cuda()]
    library_names = ["constant", "constant"]

    z_combined = torch.concat([z, dz], 1)

    for i in range(2*latent_dim):
        library.append(z_combined[:,i])
        library_names.append("z_combined["+str(i)+"]")

    if poly_order > 1:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                library.append(torch.multiply(z_combined[:,i], z_combined[:,j]))
                library_names.append("z_combined["+str(i)+"]*"+"z_combined["+str(j)+"]")

    if poly_order > 2:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k])

    if poly_order > 3:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p])

    if poly_order > 4:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        for q in range(p,2*latent_dim):
                            library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p]*z_combined[:,q])

    if include_sine:
        for i in range(2*latent_dim):
            library.append(torch.sin(sine_k*z_combined[:,i]))
            library_names.append("sin(z_combined["+str(i)+"])")
    
    if print_names == True:
        print(library_names)

    return torch.stack(library, axis=1)

def e_sindy_library_torch_order2(z, dz, latent_dim, poly_order, include_sine=False, sine_k = 1.0, print_names = "False"):
    """
    Build the SINDy library for a second order system. This is essentially the same as for a first
    order system, but library terms are also built for the derivatives.
    """
    library = [torch.ones(z.shape[0]).cuda()]
    library_names = ["constant", "constant"]

    z_combined = torch.concat([z, dz], 1)

    for i in range(2*latent_dim):
        library.append(z_combined[:,i])
        library_names.append("z_combined["+str(i)+"]")

    if poly_order > 1:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                library.append(torch.multiply(z_combined[:,i], z_combined[:,j]))
                library_names.append("z_combined["+str(i)+"]*"+"z_combined["+str(j)+"]")

    if poly_order > 2:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k])

    if poly_order > 3:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p])

    if poly_order > 4:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        for q in range(p,2*latent_dim):
                            library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p]*z_combined[:,q])

    if include_sine:
        for i in range(2*latent_dim):
            library.append(torch.sin(sine_k*z_combined[:,i]))
            library_names.append("sin(z_combined["+str(i)+"])")
    
    if print_names == True:
        print(library_names)

    return torch.stack(library, axis=1)

def sindy_library_torch_double_pendulum(z, dz, ddz, latent_dim, poly_order, include_sine=False, sine_k = 1.0, print_names = "False"):
    """
    Build the SINDy library for a second order system. This is essentially the same as for a first
    order system, but library terms are also built for the derivatives.
    """
    library = [torch.ones(z.shape[0]).cuda()]
    library_names = ["constant", "constant"]

    z_combined = torch.concat([z, dz], 1)

    for i in range(2*latent_dim):
        library.append(z_combined[:,i])
        library_names.append("z_combined["+str(i)+"]")

    if poly_order > 1:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                library.append(torch.multiply(z_combined[:,i], z_combined[:,j]))
                library_names.append("z_combined["+str(i)+"]*"+"z_combined["+str(j)+"]")

    if poly_order > 2:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k])

    if poly_order > 3:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p])

    if poly_order > 4:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        for q in range(p,2*latent_dim):
                            library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p]*z_combined[:,q])

    if include_sine:
        for i in range(2*latent_dim):
            library.append(torch.sin(sine_k*z_combined[:,i]))
            library_names.append("sin(z_combined["+str(i)+"])")
    
    if print_names == True:
        print(library_names)

    return torch.stack(library, axis=1)

def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l