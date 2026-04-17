import numpy as np


def morisita_index(points,
                   region=None,
                   max_divisions=20,
                   return_diameter=False):
    """
    Compute the Morisita index, indicating the ratio of point pairs that
    fall into the same quadrant of the region versus different quadrants.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        The points to compute the Morisita index for.

    region : numpy.ndarray[d,2], optional
        The corners of the rectangular region, given as the minimum and
        maximum values for each coordinate, eg. ``[[x_min, xmax], [y_min, y_max], ...]``.

        If not provided, will be assumed to be the exact space that the
        points occupy.

    max_divisions : int
        The maximum number of divisions for each coordinate to use in
        computing the index. Note that this isn't the number of total
        subdivisions in the space, which will be this number raised to
        the dimensionality of the system.

    return_diameter : bool
        Whether to return the number of quadrat divisions along one axis
        (default, False) or the diameter of the quadrats (True) alongside
        the index.

    Returns
    -------
    x_arr : numpy.ndarray[M]
        Array of the independent variable for the calculation, either the
        number of quadrats or the diameter of quadrats, depending on the
        value of ``return_diameter``.

    index_arr : numpy.ndarray[M]
        Array of the Morisita index values.

    """
   
    n, d = np.shape(points)

    if not hasattr(region, '__iter__'):
        # Compute the corners of the region
        bounds = np.array([np.min(points, axis=0), np.max(points, axis=0)]).T

    else:
        assert np.shape(region) == (d, 2), f'Invalid region provided; should have shape {(d, 2)} for given points.'

        bounds = np.array(region)

    # The number of subdivisions of the space
    divisions_arr = np.arange(2, max_divisions)
    index_arr = np.zeros(len(divisions_arr))

    # Transform the points such that one corner of the space is at the origin.
    adj_points = points - bounds[:,0]
    region_size = bounds[:,1] - bounds[:,0]

    total_ordered_pairs = n * (n - 1)

    for i, m in enumerate(divisions_arr):
        quadrat_size = region_size / m

        quadrat_identities = np.floor(adj_points / quadrat_size).astype(np.int64)

        # An easy way to calculate the number of pairs that share the same
        # quadrat is to compute the number of unique quadrat coordinates.
        # We don't care about which quadrat has how many points, so we can
        # just store the counts.
        _, quadrat_point_counts = np.unique(quadrat_identities, return_counts=True, axis=0)
        quadrat_pairs = np.sum([p * (p - 1) for p in quadrat_point_counts])

        # If there are no pairs in any quadrat, we certainly won't have more by
        # subdividing further, so we can exit now.
        if quadrat_pairs == 0:
            break

        # m to the power d since that is how many quadrats there actually
        # are in the d-dimensional space.
        index_arr[i] = m**d * quadrat_pairs / total_ordered_pairs

    if return_diameter:
        diameter_arr = np.array([np.linalg.norm(region_size / m) for m in divisions_arr])
        return diameter_arr, index_arr

    return divisions_arr**d, index_arr


def quadrat_iod(points,
                region=None,
                max_divisions=20,
                return_diameter=False):
    r"""
    Compute the index of dispersion for quadrat counts as a function of
    the number of quadrats the space is divided into.

    This metric is directly related to the Morisita index, M, through the
    following relation:

    .. math::

        I = \frac{m}{m - 1} \left[ (n - 1) M - (n/m - 1) \right]

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        The points to compute the Morisita index for.

    region : numpy.ndarray[d,2], optional
        The corners of the rectangular region, given as the minimum and
        maximum values for each coordinate, eg. ``[[x_min, xmax], [y_min, y_max], ...]``.

        If not provided, will be assumed to be the exact space that the
        points occupy.

    max_divisions : int
        The maximum number of divisions for each coordinate to use in
        computing the index. Note that this isn't the number of total
        subdivisions in the space, which will be this number raised to
        the dimensionality of the system.

    return_diameter : bool
        Whether to return the number of quadrat divisions along one axis
        (default, False) or the diameter of the quadrats (True) alongside
        the index.

    Returns
    -------
    x_arr : numpy.ndarray[M]
        Array of the independent variable for the calculation, either the
        number of quadrats or the diameter of quadrats, depending on the
        value of ``return_diameter``.

    index_arr : numpy.ndarray[M]
        Array of the quadrat index of dispersion values.
    """
    
    n, d = np.shape(points)

    if not hasattr(region, '__iter__'):
        # Compute the corners of the region
        bounds = np.array([np.min(points, axis=0), np.max(points, axis=0)]).T

    else:
        assert np.shape(region) == (d, 2), f'Invalid region provided; should have shape {(d, 2)} for given points.'

        bounds = np.array(region)

    # The number of subdivisions of the space
    divisions_arr = np.arange(2, max_divisions)
    index_arr = np.zeros(len(divisions_arr))

    # Transform the points such that one corner of the space is at the origin.
    adj_points = points - bounds[:,0]
    region_size = bounds[:,1] - bounds[:,0]

    total_ordered_pairs = len(points) * (len(points) - 1)

    for i, m in enumerate(divisions_arr):
        quadrat_size = region_size / m

        quadrat_identities = np.floor(adj_points / quadrat_size).astype(np.int64)

        # An easy way to calculate the number of pairs that share the same
        # quadrat is to compute the number of unique quadrat coordinates.
        # We don't care about which quadrat has how many points, so we can
        # just store the counts.
        _, quadrat_point_counts = np.unique(quadrat_identities, return_counts=True, axis=0)

        # This is the (sample) variance divided by the mean.
        # ddof tells numpy to calculate the sample variance, ie. using m**d - 1
        # in the denominator instead of m**d. 
        index_arr[i] = np.var(quadrat_point_counts, ddof=1) / (n / m**d)

    if return_diameter:
        diameter_arr = np.array([np.linalg.norm(region_size / m) for m in divisions_arr])
        return diameter_arr, index_arr

    return divisions_arr**d, index_arr


#def K(points,
