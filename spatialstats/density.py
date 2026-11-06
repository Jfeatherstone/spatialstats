import numpy as np
from scipy.signal import convolve

def density_field(points,
                 values=None,
                 lattice_spacing=None,
                 field_dims=None,
                 fixed_bounds=None,
                 kernel='gaussian',
                 gaussian_width=1,
                 kernel_size=None,
                 return_axes=False):
    """
    Compute the density field for a set of points.

    Unless ``values`` is provided, course-grained field is the point density.

    This is a relatively generalized way of computing a density field using
    a Fourier transform; it will likely be much more computationally
    intensive than, for example, just rescaling points and using rounding
    to bin them. That being said, you get much more control over the
    parameters of the field calculation.

    Parameters
    ----------
    points : numpy.ndarray[N, d]
        Spatial positions of N points in d-dimensional space.

    values : numpy.ndarray[N, [k]] or func(points)->numpy.ndarray[N, [k]], optional
        Field values at each point. Can be k-dimensional vector,
        resulting in k course-grained fields.

        Can also be a (vectorized) function that returns a value given
        a collection of points. eg. neighbor counting function. This
        functionality is provided such that if the function is computationally
        expensive, eg. neighbor counting, the points can be subdivided into
        batches and the course grained fields can be summed at the end. This
        is a way to approximate the course grained field for a huge (>1e6)
        number of points, while still remaining computationally feasible.

        If not provided, returned field will be the point density.

    lattice_spacing : float, optional
        The spacing of lattice points for the course-grained field.

        If not provided, will be chosen such that every dimension
        has 100 lattice points (implying potentially hetereogeneous
        spacing across dimensions).

    field_dims : array_like[d], optional
        The size of the lattice to create; alternative option instead
        of giving a specific spacing to define the lattice.

        If a lattice spacing is given, this overrides the field size.

    fixed_bounds : array_like[d,2], optional
        The bounds of the field to define the discretized
        grid over. If not provided, will be calculated based on the
        extrema of the provided set of points. Should be given in the
        format:

            [[x_min, x_max], [y_min, y_max], ...]

    kernel : {'gaussian', 'ones'} or numpy.ndarray[A, A]
        The kernel to course-grain the field with. ``'gaussian'`` option is
        implemented as default, but a custom matrix can be provided. If using
        default gaussian option, kernel size can be set with ``kernel_size``.
        ``'ones'`` will create a square matrix of 1 values.

    kernel_size : int, optional
        The kernel size to use if ``kernel`` is ``'gaussian'`` or ``'ones'``.
        If a custom kernel is provided, this has no effect.

        If using a Gaussian kernel, and this value is not specified, a suitable
        value will be selected such that the Gaussian kernel decays to at least
        1e-4 at the edge.

    return_axes : bool
        Whether to return the axes defining the points on the lattice alongside
        the field.

    Returns
    -------
    field : numpy.ndarray[n,m,...,[k]]
        Course grained field, with `d` or `d+1` dimensions, depending on
        if the field was course grained based on point density (`d`),
        or based on vector values (`d+1`).

    axes : (numpy.ndarray[n], numpy.ndarray[m], ...), optional
        The axis for each dimension. Only returned
        if ``return_axes=True``.

    Notes
    -----
    The field calculation involves using fourier transforms to compute
    the convolution of the given kernel, which often leads to zero-value
    regions having some nonzero value on the order of machine precision
    (~1e-15). If you plan to use the result in an expression of the sort:

        np.where(field > 0)

    or something similar, it is highly recommended to clamp tiny values
    or use a proper threshold:

        field[field < 1e-10] = 0
        ...
        np.where(field > 1e-10)
    """
    # TODO: Make sure this works for 1D data
    dim = np.shape(points)[-1] if len(np.shape(points)) > 1 else 1

    if dim == 1:
        points = np.array(points)[:, None]
    
    if not hasattr(fixed_bounds, '__iter__'):
        occupied_volume_bounds = np.array(list(zip(np.min(points, axis=0), np.max(points, axis=0))))
    else:
        occupied_volume_bounds = np.array(fixed_bounds)

    # Add a small epsilon so we don't round up over the lattice size
    epsilon = 1e-6
    
    # Create a lattice with the selected scale for that cube
    if lattice_spacing is not None:
        spacing = lattice_spacing
        # We also have to correct the occupied volume bounds if we were provided with
        # a fixed set of bounds. Otherwise, we will end up with an extra bin at the
        # end
        if hasattr(fixed_bounds, '__iter__'):
            occupied_volume_bounds[:, 1] -= spacing

    elif lattice_spacing is None and hasattr(field_dims, '__iter__'):
        # If we are given a specific size of the lattice, we use that
        # And the -1 is because we later add 1 
        spacing = (occupied_volume_bounds[:, 1] - occupied_volume_bounds[:, 0]) / (np.array(field_dims) - 1) + epsilon

    else:
        # If we are given nothing, we choose such that the every axis has
        # 100 lattice points (arbitrary, but reasonable)
        spacing = (occupied_volume_bounds[:, 1] - occupied_volume_bounds[:, 0]) / 100

    # Calculate the actual field dims, which should be the same as field_dims
    # if it was provided, but is needed if the lattice is specified by a
    # spacing.
    # I was having some index out of bounds issues, so I think adding a tiny epsilon
    # to the range of the field helps resolve that. Same with the +1
    true_field_dims = np.ceil((occupied_volume_bounds[:, 1] - occupied_volume_bounds[:, 0] + epsilon) / (spacing))
    true_field_dims = true_field_dims.astype(np.int64)

    # This will be a list of arrays that have length true_field_dims + 1
    bin_edges = [occupied_volume_bounds[i,0] + spacing * np.arange(true_field_dims[i] + 1) for i in range(dim)]
    # These are the centers of each bin, ie. the coordinates that we will
    # associated with the value at each lattice point.
    axes = [(b[1:] + b[:-1]) / 2 for b in bin_edges]

    # Calculate which lattice cell each scatter point falls into
    lattice_positions = np.floor((points - occupied_volume_bounds[:,0]) / spacing).astype(np.int64)

    # Find all points that are in the relevant region
    in_bounds = np.array([(lattice_positions[:,i] >= 0) & (lattice_positions[:,i] < true_field_dims[i]) for i in range(dim)]).T
    good_indices = np.where([ib.all() for ib in in_bounds])[0]

    # Check if an array of values was passed for each point
    # Otherwise we just have a scalar field (and we'll collapse
    # the last dimension later on).
    if hasattr(values, '__iter__'):
        k = np.shape(values)[-1]
        val_arr = values

    else:
        k = 1
        val_arr = np.ones((np.shape(points)[0], 1))

    field_arr = np.zeros((*true_field_dims, k))

    # Instead of actually applying a gaussian kernel now, which would be
    # very inefficient since we'd need to sum a potentially very large number
    # of k*d dimensional matrices (more or less), we instead just assign each
    # lattice point, then smooth over it after with the specified kernel.
    # Where this might cause issues:
    # - If the lattice spacing is too large, you will get some weird artifacts
    #   from this process. Though in that case, you'll get a ton of artifacts from
    #   elsewhere too, so just don't use too large a lattice spacing :)

    for i in good_indices:
        field_arr[tuple(lattice_positions[i])] += val_arr[i]

    # Now smooth over the field
    if kernel == 'gaussian':

        if kernel_size is None:
            # Otherwise, calculate when the gaussian kernel would reach the
            # level of 1e-5 (a bit arbitrary, but reasonable).
            eps = 1e-5
            distance = np.sqrt(-np.log(eps)) * 2 * gaussian_width
            # Convert to grid units
            kernel_size = int(np.ceil(distance / spacing))

        single_axis = np.arange(kernel_size)

        # This creates a grid of numbers like (a . represents decimals I'm not showing)
        # 3. 1. 2  1. 3.
        # 2. 1. 1  1. 2.
        # 2  1  0  1  2
        # 2. 1. 1  1. 2
        # 3. 1. 2  2. 3
        kernel_grid = np.array(np.meshgrid(*np.repeat([single_axis], np.shape(points)[-1], axis=0))) - (kernel_size - 1) / 2
        # Now scale it back to the scale of the data
        distance_arr = np.sum([(kernel_grid[i] * spacing)**2 for i in range(np.shape(points)[-1])], axis=0)
        # Compute the Gaussian
        kernel_arr = np.exp(-distance_arr / (np.sqrt(2) * gaussian_width)**2)

    elif kernel == 'ones':
        kernel_arr = np.ones([kernel_size for _ in range(dim)])

    # Otherwise, we expect that kernel should already be passed as a
    # proper square d-dimensional matrix
    else:
        kernel_arr = kernel

    # Perform a convolution of the field with our kernel
    # 'same' keeps the same bounds on the field, but might cause
    # some weird effects near the boundaries
    trans_convolution = np.zeros_like(field_arr.T)

    for i in range(k):
        # Note that convolve(x, y) == convolve(x.T, y.T).T
        # We need this so we can go over our k axis
        trans_convolution[i] = convolve(field_arr.T[i], kernel_arr.T, mode='same') / np.sum(kernel_arr)

    convolution = trans_convolution.T

    # If k == 1, collapse the extra dimension
    if k == 1:
        convolution = convolution[..., 0]
    
    if return_axes:
        return convolution, tuple(axes)

    return convolution
