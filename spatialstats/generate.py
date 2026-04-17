import numpy as np

import warnings
from scipy.special import gamma
from scipy.spatial import KDTree

def generate_square_lattice(N,
                            d=2,
                            region=None):
    """
    Generate a set of points distributed in a square lattice.

    The lattice spacing will depend on the region size (if provided); if you
    provide an anisotropic region, the lattice spacing will also be anisotropic.

    Parameters
    ----------
    N : int
        The number of points to generate. Should be a perfect square, cube,
        etc., ie. has an integer value when raised to the power ``1/d``, 
        otherwise not all of the points will be located in the region.

    d : int
        The dimensionality of the space to generate points in.

    region : numpy.ndarray[d,2], optional
        The corners of the rectangular region, given as the minimum and
        maximum values for each coordinate, eg. ``[[x_min, xmax], [y_min, y_max], ...]``.

        If not provided, points will be generated in the unit hypercube of
        dimension given by ``d``, ie. between [0, 1] for each coordinate.

    Returns
    -------
    points : numpy.ndarray[N,d]
        The generated points.
    """
    if not hasattr(region, '__iter__'):
        region = np.array([[0, 1]]*d)

    else:
        region = np.array(region)

    assert len(region) == d, f'Incompatible region ({region}) and dimensionality ({d}) provided.'

    if np.around(N**(1/d)) - N**(1/d) > 1e-8:
        raise ValueError(f'Number of points ({N}) doesn\'t fit nicely into a a {d}-dimensional space!')

    axes = [np.arange(N**(1/d))]*d
    points = np.array(np.meshgrid(*axes)).T

    # The double exponentiation here will cause round off issues if you
    # don't provide a value of N that is a perfect square/cube/etc.
    points = points.reshape((int(np.around(N**(1/d)))**d, d))

    # Now scale up to the lattice size

    # Compute the size of the region, so we can calculate the lattice spacing.
    axis_sizes = region[:,1] - region[:,0]
    lattice_spacing = axis_sizes / N**(1/d)

    # Add half the lattice spacing so the points are centered
    points = points * lattice_spacing + lattice_spacing / 2 + region[:,0]

    return points


def generate_tri_lattice(N,
                         d=2,
                         region=None):
    """
    Generate a set of points distributed in a square lattice.

    The lattice spacing will depend on the region size (if provided); if you
    provide an anisotropic region, the lattice spacing will also be anisotropic.

    Parameters
    ----------
    N : int
        The number of points to generate. Should be a perfect square, cube,
        etc., ie. has an integer value when raised to the power ``1/d``, 
        otherwise not all of the points will be located in the region.

    d : int
        The dimensionality of the space to generate points in.

    region : numpy.ndarray[d,2], optional
        The corners of the rectangular region, given as the minimum and
        maximum values for each coordinate, eg. ``[[x_min, xmax], [y_min, y_max], ...]``.

        If not provided, points will be generated in the unit hypercube of
        dimension given by ``d``, ie. between [0, 1] for each coordinate.

    Returns
    -------
    points : numpy.ndarray[N,d]
        The generated points.
    """

    if d > 3:
        raise NotImplementedError('Only implemented for d <= 3.')

    if not hasattr(region, '__iter__'):
        region = np.array([[0, 1]]*d)

    else:
        region = np.array(region)

    assert len(region) == d, f'Incompatible region ({region}) and dimensionality ({d}) provided.'

    if np.around(N**(1/d)) - N**(1/d) > 1e-8:
        raise ValueError(f'Number of points ({N}) doesn\'t fit nicely into a a {d}-dimensional space!')

    # Compute the size of the region, so we can calculate the lattice spacing.
    axis_sizes = region[:,1] - region[:,0]
    lattice_spacing = axis_sizes / N**(1/d)

    # Start with a square lattice
    points = generate_square_lattice(N, d, region)

    # Now displace the points to form a triangular lattice
    # TODO: This might not be totally correct...
    n_row = int(np.around(N**(1/d)))
    for k in range(d-1):
        for i in range(n_row):
            for j in range(n_row):
                points[n_row**2*i + j*n_row:n_row**2*i + (j+1)*n_row, k+1] += (j%2)*lattice_spacing[k]/2 - lattice_spacing[k]/4 

    return points


def generate_independent(N,
                         d=2,
                         region=None,
                         exclusion=None):
    """
    Generate a set of points distributed independently of each other.

    Parameters
    ----------
    N : int
        The number of points to generate. Should be a perfect square, cube,
        etc., ie. has an integer value when raised to the power ``1/d``, 
        otherwise not all of the points will be located in the region.

    d : int
        The dimensionality of the space to generate points in.

    region : numpy.ndarray[d,2], optional
        The corners of the rectangular region, given as the minimum and
        maximum values for each coordinate, eg. ``[[x_min, xmax], [y_min, y_max], ...]``.

        If not provided, points will be generated in the unit hypercube of
        dimension given by ``d``, ie. between [0, 1] for each coordinate.

    exclusion : float
        The 'size' of the points; the resulting set of points will be
        generated such that no two points are closer together than this
        exclusion size.

        It can be difficult to generate points that encompass a large
        fraction of the space; this function will iteratively add independently-
        chosen points iteratively to reach the total number. If after
        100 iterations, the requested number of points is not reached,
        this function will raise an error. This is because if you are
        generating enough points to need to constantly rejection-sample,
        you will likely end up with a set of correlated points, even if
        the initial positions are chosen independently (since you're
        rejecting the independent points based on radial distances).

    Returns
    -------
    points : numpy.ndarray[N,d]
        The generated points.
    """

    if not hasattr(region, '__iter__'):
        region = np.array([[0, 1]]*d)

    else:
        region = np.array(region)

    assert len(region) == d, f'Incompatible region ({region}) and dimensionality ({d}) provided.'

    # If there is no exclusion, we can just generate a uniform sample of
    # points from numpy and be done.
    if exclusion is None:
        points = np.array([np.random.uniform(r[0], r[1], size=N) for r in region]).T
        return points

    # Otherwise, we are going to have to draw points sequentially to make
    # sure there isn't any overlap.

    axis_sizes = region[:,1] - region[:,0]

    # Make sure it seems possible to fit this many points in the area
    occupied_volume = np.pi**(d/2) / gamma(d/2 + 1) * exclusion**d * N

    if occupied_volume >= np.prod(axis_sizes):
        raise ValueError(f"Total area of points ({occupied_volume}) would exceed available space in region ({np.prod(axis_sizes)}).")

    # Print a warning if you are close, just in case
    if occupied_volume >= np.prod(axis_sizes) * 0.1:
        warnings.warn('Occupied volume of points is a macroscopic fraction of full space, generation might fail.')

    # Max 10 tries to generate the traps
    # This is chosen empirically, and the specific value isn't important.
    # We build this list iteratively.
    points = []

    for i in range(100):
        new_points = np.array([np.random.uniform(r[0], r[1], size=N-len(points)) for r in region]).T

        # We'll use a kdtree to make sure that the points aren't too close to
        # each other.
        kdtree = KDTree(points + list(new_points))

        # Find all points within the twice the exclusion distance for each point.
        nearby_indices = kdtree.query_ball_tree(kdtree, exclusion*2)

        # Now we run through all of the points and figure out which ones
        # are bad. Note that we might mark new point i as bad, which would
        # then mean point j becomes good since it was only close to point i.
        bad_indices = []

        # We don't need to check the old points
        for j in range(len(new_points)):
            neighbors = nearby_indices[len(points) + j]

            # Remove all points that we have already removed, as well as the
            # point itself.
            neighbors = [n for n in neighbors if (n not in bad_indices) and (n != (len(points) + j))]

            if len(neighbors) > 0:
                bad_indices += [len(points) + j]


        # Remove all of the bad points and add the good ones to the
        # running list.
        new_points = [new_points[j] for j in range(len(new_points)) if not (len(points) + j) in bad_indices]

        points += new_points

        if len(points) == N:
            break

    points = np.array(points)

    if len(points) != N:
        raise ValueError(f'Unable to generate full amount of points ({len(points)} / {N}) due to exclusion constraint.')

    return points


def generate_regular(N,
                     region=(1, 1), regularity=0, trapSize=.01):

    # Make sure it seems possible to fit this many traps in the area
    assert N*np.pi*trapSize**2 < boxSize[0]*boxSize[1], "Total area of traps would exceed available space"

    # Our positions start as a triangular lattice
    initialPositions = generateTrapsTriLattice(N, boxSize, trapSize)

    if disorder == 0:
        return initialPositions

    # Max 100 tries to generate the traps
    for i in range(100):
        perturbationLengthScale = disorder * np.sqrt(boxSize[0]**2 + boxSize[1]**2)
        displacementFromLattice = np.random.uniform(0, perturbationLengthScale, size=(N, 2))

        # Initial positions plus the displacement scaled by the disorder factor
        newPositions = initialPositions + displacementFromLattice

        # Mod positions so the traps always stay in the box
        newPositions[:,0] = newPositions[:,0] % boxSize[0]
        newPositions[:,1] = newPositions[:,1] % boxSize[1]

        # Remove overlaps within new points
        distMat = np.array([[np.sum((newP1 - newP2)**2) for newP1 in newPositions] for newP2 in newPositions])
        # Remove diagonal elements
        distMat[np.arange(N),np.arange(N)] = np.inf

        # Find points that are close to each other
        closePoints = np.where(distMat < 2*trapSize)[0]

        if len(closePoints) > 0:
            continue
        
        return newPositions

    raise Exception('Couldn\'t generate trap configuration after 100 tries.') 

