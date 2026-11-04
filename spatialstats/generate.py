import numpy as np

import warnings
from scipy.special import gamma
from scipy.spatial import KDTree
from scipy.optimize import bisect

from shapely.geometry import Polygon, Point

import matplotlib.pyplot as plt

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
    Generate a set of points distributed in a triangular lattice.

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


def generate_poisson(N, d=2, region=None, diameter=None, density=None):
    """
    Generate a set of points distributed independently of each other.

    When given a value for ``diameter``, ie. an effective size of the
    points with the constraint that they cannot overlap, selection is
    done sequentially via rejection. Points are randomly generated within
    the relevant region, and then if they do not overlap with the existing
    points, they are kept. No spatial optimization/rejection is performed
    to maximize the efficiency of generation.


    Parameters
    ----------
    N : int
        The number of points to generate. Should be a perfect square, cube,
        etc., ie. has an integer value when raised to the power ``1/d``, 
        otherwise not all of the points will be located in the region.

    d : int
        The dimensionality of the space to generate points in.

    region : numpy.ndarray[d,2] or shapely.geometry.Polygon, optional
        The corners of the rectangular region, given as the minimum and
        maximum values for each coordinate, eg. ``[[x_min, xmax], [y_min, y_max], ...]``.

        For two-dimensional generation, can also be given as a Shapely polygon,
        which allows for more complex shapes than simple rectangles.

        If not provided, points will be generated in the unit hypercube of
        dimension given by ``d``, ie. between [0, 1] for each coordinate.

    diameter : float
        The diameter of the points; the resulting set of points will be
        generated such that no two points are closer together than this
        distance, given in the same units as the region.

        It can be difficult to generate points that encompass a large
        fraction of the space; this function will iteratively add independently-
        chosen points to reach the total number. If after 100 iterations,
        the requested number of points is not reached, this function will
        raise an error. This is because if you are generating enough points
        to need to constantly rejection-sample, you will likely end up with
        a set of correlated points, even if the initial positions are chosen
        independently (since you're rejecting the independent points based
        on radial distances).

    density : (numpy.ndarray[m,n,...], (numpy.ndarray[m], numpy.ndarray[n], ...)), optional
        The density field for the process if point generation is to be
        inhomogeneous. Should be given as a tuple including the multidimensional
        histogram values, and an array giving the bin centers for each axis.


    Returns
    -------
    numpy.ndarray[N,d]
        The generated points.


    Examples
    --------
    Generate homogeneous Poisson points in a rectangle using a Shapely
    boundary.

        import numpy as np
        from shapely.geometry import Polygon
        from spatialstats import generate_poisson

        # Define the boundary, the ordered set of boundary points
        xmax, ymax = 5, 5
        boundary = np.array([[0, 0],
                             [0, ymax],
                             [xmax, ymax],
                             [xmax, 0]])

        poly = Polygon(boundary)

        points = generate_poisson(100, region=poly)

    Generate inhomogeneous Poisson points based off of an existing set of
    points.

        import numpy as np
        from shapely.geometry import Polygon
        from spatialstats import generate_poisson, density_field

        # Random data just to demonstrate
        prior_points = np.random.uniform(0, 1, size=(1000,2))

        # Compute density field
        density, axes = density_field(prior_points, return_axes=True)

        points = generate_poisson(100, density=(density, axes))
    """
    # Since we may be given a complex, potentially concave region (in 2d), we
    # generate points in two phases. First, we identify the smallest
    # rectangle that fits the proper region, and generate points in that
    # space. Then, we throw away points that are outside of the true region.
    # Depending on how concave your true region is, this can get quite
    # inefficient, just a warning.

    # For all other dimensions, we don't need such complexity.

    # If we are generating traps inhomogeneously, we need to parse the density
    # field data.
    homogeneous = True
    if hasattr(density, '__iter__'):
        homogeneous = False

        # The format is (hist, (ax1, ax2, ...))
        density_arr, axes = density

        # Normalize
        density_arr = density_arr / np.sum(density_arr)

        # This is what we will need for the actual random generation
        cumulative_density_flat = np.cumsum(density_arr.flatten())

        # The axes have the same length as each dimension of the density
        # field, so they represent the lattice sites, or equivalently, the
        # center of the bins.
        # It is useful to have the bin edges as well.
        axis_spacing = np.array([(ax[1] - ax[0]) for ax in axes])
        bin_edges = [np.zeros(len(ax)+1) for ax in axes]

        for i in range(d):
            bin_edges[i][0] = axes[i][0] - axis_spacing[i]/2
            bin_edges[i][1:] = axes[i] + axis_spacing[i]/2


    if not hasattr(region, '__iter__') and type(region) is not Polygon:
        # If we aren't given a region but we are given a density map, we
        # can use the region that the density map is defined to set our
        # region of interest.
        if not homogeneous:
            # Add some small epsilon to max sure we are in bounds; this is
            # to prevent machine rounding errors that lead to the function
            # thinking the region is wider than the provided density map.
            region_min = np.array([np.min(bin_edges[i]) for i in range(d)]) + 1e-10
            region_max = np.array([np.max(bin_edges[i]) for i in range(d)]) - 1e-10

            region = np.array(list(zip(region_min, region_max)))

        else:
            # Otherwise, generate points in the unit square.
            region = np.array([[0, 1]]*d)

        inner_region = None

    elif type(region) is Polygon and d == 2:
        # If we are given a polygon, we already have our proper region,
        # then we just need to generate the outer one.
        inner_region = region
        coords = region.exterior.xy
        region = np.array([(np.min(coords[i], axis=0), np.max(coords[i], axis=0)) for i in range(d)])

    elif hasattr(region, '__iter__'):
        region = np.array(region)
        inner_region = None

    else:
        raise ValueError('Invalid region and/or dimensionality provided!')

    assert len(region) == d, f'Incompatible region ({region}) and dimensionality ({d}) provided.'

    # If we have have imhomogeneous sampling and have a defined region, make,
    # sure that the density field covers the entire region.
    if not homogeneous:
        for i in range(d):
            assert (region[i,0] >= bin_edges[i][0]) and (region[i,1] <= bin_edges[i][-1]), 'Density field doesn\'t cover entire region.'

    axis_sizes = region[:,1] - region[:,0]

    # Make sure it seems possible to fit this many points in the area
    if diameter is not None:
        occupied_volume = np.pi**(d/2) / gamma(d/2 + 1) * (diameter/2)**d * N

        if occupied_volume >= np.prod(axis_sizes):
            raise ValueError(f"Total area of points ({occupied_volume}) would exceed available space in region ({np.prod(axis_sizes)}).")

        # Print a warning if you are close, just in case
        if occupied_volume >= np.prod(axis_sizes) * 0.1:
            warnings.warn('Occupied volume of points is a macroscopic fraction of full space, generation might fail.')

    # We build this list iteratively.
    points = []

    # Max 100 tries to generate the traps
    # This is chosen empirically, and the specific value isn't important.
    MAX_TRIES = 100
    for i in range(MAX_TRIES):
        if homogeneous:
            new_points = np.array([np.random.uniform(r[0], r[1], size=N-len(points)) for r in region]).T

        else:
            random_values = np.random.uniform(0, 1, size=N-len(points))
            # The argmax is just finding the first index where the conditional
            # is true.
            flat_indices = np.array([np.argmax(cumulative_density_flat > r) for r in random_values])
            indices = np.unravel_index(flat_indices, density_arr.shape)

            lattice_points = np.array([axes[j][indices[j]] for j in range(d)]).T
            
            # For structure less than the length scale of the density arr
            # lattice, we just assume homogeneous.
            lattice_perturbation = np.array([np.random.uniform(-s, s, size=N-len(points)) for s in axis_spacing]).T

            new_points = lattice_points + lattice_perturbation

            # We should make sure that the points are properly in the region,
            # since there is a small chance they could end up outside.
            good_indices = []
            for j in range(len(new_points)):
                good = True
                for k in range(d):
                    if (new_points[j,k] < region[k][0]) or (new_points[j,k] > region[k][1]):
                        good = False

                if good:
                    good_indices.append(j)

            new_points = new_points[good_indices]

        # Check the inner region if necessary
        if inner_region is not None:
            new_points = np.array([p for p in new_points if inner_region.contains(Point(p))])

        # Check the exclusion condition if necessary
        if diameter is not None:

            # We'll use a kdtree to make sure that the points aren't too close to
            # each other.
            kdtree = KDTree(points + list(new_points))

            # Find all points within the twice the exclusion distance for each point.
            nearby_indices = kdtree.query_ball_tree(kdtree, diameter)

            # Now we run through all of the points and figure out which ones
            # are bad. Note that we might mark new point i as bad, which would
            # then mean point j becomes good since it was only close to point i.
            # So we need to keep track of which indices we are getting rid
            # of. This does mean there is some sensitivity to the order in
            # which we remove points; we don't optimize here to remove the
            # least amount of points or anything, just sequentially.
            bad_indices = []

            # We don't need to check the old points
            for j in range(len(new_points)):
                neighbors = nearby_indices[len(points) + j]

                # Remove all points that we have already removed, as well as the
                # point itself.
                neighbors = [n for n in neighbors if (n not in bad_indices) and (n != (len(points) + j))]

                if len(neighbors) > 0:
                    bad_indices += [len(points) + j]


            # Remove all of the bad points
            new_points = np.array([new_points[j] for j in range(len(new_points)) if not (len(points) + j) in bad_indices])

        # Add the new points to the running list
        points += list(new_points)

        if len(points) == N:
            break

    points = np.array(points)

    if len(points) != N:
        raise ValueError(f'Unable to generate full amount of points ({len(points)} / {N}) due to exclusion constraint.')

    return points


def generate_cluster(parent_mean=1, parent_process='poisson', parent_density=None, child_mean=1, child_process='gaussian', child_process_scale=1, region=None, diameter=None, soft_diameter=None, use_parent_buffer=True, generation_order='parent', debug=False):
    """
    Generate a set of points through a Neyman-Scott Cox cluster process.

    Generation is done in two phases: first, 'parent' points are generated,
    which are used as the centers around which 'child' points are generated.
    The set of all child points is the returned point pattern.

    When given a value for ``diameter``, ie. an effective size of the
    points with the constraint that they cannot overlap, selection is
    done sequentially via rejection. Points are randomly generated within
    the relevant region, and then if they do not overlap with the existing
    points, they are kept. No spatial optimization/rejection is performed
    to maximize the efficiency of generation.


    Parameters
    ----------
    parent_mean : float
        The expectation (and variance) of the number of parent points to
        generate. Random value is drawn from a Poisson distribution.

    parent_process : {'poisson'} or func(N, region, density)
        The process used to generate the parent points. Defaults to Poisson
        (independent) generation.

        Can also be provided as a function that takes the following arguments
        and returns an array of points:

            N : int
                The number of points to generate.

            d : int
                The dimension to generate points in.

            region : shapely.geometry.Polygon
                The region within which points are to be generated.

            density : (numpy.ndarray[m,n], (numpy.ndarray[m], numpy.ndarray[n])), optional
                The density field for the process if point generation is to be
                inhomogeneous. Should be given as a tuple including the multidimensional
                histogram values, and an array giving the bin centers for each axis.

    parent_density : (numpy.ndarray[m,n], (numpy.ndarray[m], numpy.ndarray[n]), optional
        The density field for the parent process if point generation is to be
        inhomogeneous. Should be given as a tuple including the multidimensional
        histogram values, and an array giving the bin centers for each axis.

    region : numpy.ndarray[2,2] or shapely.geometry.Polygon, optional
        The corners of the rectangular region, given as the minimum and
        maximum values for each coordinate, eg. ``[[x_min, xmax], [y_min, y_max]]``.

        For two-dimensional generation, can also be given as a Shapely polygon,
        which allows for more complex shapes than simple rectangles.

        If not provided, points will be generated in the unit square ie. between
        [0, 1] for each coordinate.

    diameter : float
        The diameter of the points; the resulting set of points will be
        generated such that no two points are closer together than this
        distance, given in the same units as the region.

        It can be difficult to generate points that encompass a large
        fraction of the space; this function will iteratively add independently-
        chosen points to reach the total number. If after 100 iterations,
        the requested number of points is not reached, this function will
        raise an error. This is because if you are generating enough points
        to need to constantly rejection-sample, you will likely end up with
        a set of correlated points, even if the initial positions are chosen
        independently (since you're rejecting the independent points based
        on radial distances).

    generation_order : {'parent', 'child', 'random'}
        The order to generate the points in; only has an effect if a non-zero
        value is given for ``diameter``.

        'parent' means parent-by-parent: all of the children for a parent
        will be generated before moving onto the next parent.

        'child' means child-by-child: one child will be generated for each
        parent, until there are no more children left to generate.

        'random' means the order is totally random.


    Returns
    -------
    numpy.ndarray[N,d]
        The generated points.


    Examples
    --------
    Generate homogeneous Poisson points in a rectangle using a Shapely
    boundary.

        import numpy as np
        from shapely.geometry import Polygon
        from spatialstats import generate_poisson

        # Define the boundary, the ordered set of boundary points
        xmax, ymax = 5, 5
        boundary = np.array([[0, 0],
                             [0, ymax],
                             [xmax, ymax],
                             [xmax, 0]])

        poly = Polygon(boundary)

        points = generate_poisson(100, region=poly)

    Generate inhomogeneous Poisson points based off of an existing set of
    points.

        import numpy as np
        from shapely.geometry import Polygon
        from spatialstats import generate_poisson, density_field

        # Random data just to demonstrate
        prior_points = np.random.uniform(0, 1, size=(1000,2))

        # Compute density field
        density, axes = density_field(prior_points, return_axes=True)

        points = generate_poisson(100, density=(density, axes))
    """
    # This only works for 2D case
    d = 2

    # Parse the region
    if not hasattr(region, '__iter__') and type(region) is not Polygon:
        # Generate points in the unit square.
        region_rect = np.array([[0, 1]]*d)

        boundary = np.array([[0, 0],
                             [0, 1],
                             [1, 1],
                             [1, 0]])

        region_poly = Polygon(boundary)
        region_rect_poly = region_poly

    elif type(region) is Polygon:
        # If we are given a polygon, we already have our proper region,
        # then we just need to generate the outer one.
        region_poly = region

        coords = region.exterior.xy
        region_rect = np.array([(np.min(coords[i], axis=0), np.max(coords[i], axis=0)) for i in range(d)])

        boundary = np.array([[region_rect[0,0], region_rect[1,0]],
                             [region_rect[0,0], region_rect[1,1]],
                             [region_rect[0,1], region_rect[1,1]],
                             [region_rect[0,1], region_rect[1,0]]])

        region_rect_poly = Polygon(boundary)

    elif hasattr(region, '__iter__'):
        region_rect = np.array(region)

        boundary = np.array([[region_rect[0,0], region_rect[1,0]],
                             [region_rect[0,0], region_rect[1,1]],
                             [region_rect[0,1], region_rect[1,1]],
                             [region_rect[0,1], region_rect[1,0]]])

        region_poly = Polygon(boundary)
        region_rect_poly = region_poly

    else:
        raise ValueError('Invalid region and/or dimensionality provided!')

    # Now we need to generate the parent region, which will be a larger
    # version of the actual region. The amount we buffer the shape
    # by will depend on the typical scale of the child process.

    if type(child_process) is str and child_process == 'uniform':
        # The child_process_scale argument gives the size of the uniform
        # disk, so that should be our buffer size.
        buffer_size = child_process_scale

        child_process_func = lambda size=1: np.random.uniform(0, child_process_scale, size=size)

    elif type(child_process) is str and child_process == 'gaussian':
        # For a gaussian kernel, we will want more than a single standard
        # deviation. 3 standard deviations gives a P value of 0.0013, which
        # seems like plenty (ie. 0.13% chance of missing points because
        # we didn't increase the buffer enough).
        buffer_size = 3 * child_process_scale

        # Divide the scale by sqrt 2 since this will generate numbers for
        # each axis offset, so the maximum distance would still be
        # child_process_scale.
        child_process_func = lambda size=1: np.random.normal(0, child_process_scale / np.sqrt(2), size=size)

    elif type(child_process) is str and child_process == 'cauchy':
        # For a cauchy kernel, we choose the distance where we have the
        # same P value as the Gaussian case (0.0013).
        sig_value = 0.0013
        buffer_size = 1 - (sig_value * 2 * np.pi * child_process_scale**2)**(3/2)

        # TODO
        raise Exception('Cauchy not implemented yet')

    elif callable(child_process):
        # If we are given a function, we assume that that function takes a
        # single argument (the distance from the parent point) and returns
        # a probability. As with the Gaussian case, we can choose a distance
        # that gives a P value of around 0.001.
        sig_value = 0.0013
        def opt_func(distance):
            return sig_value - child_process(distance)

        # We use a max of 5 standard deviations/scale factors.
        buffer_size = bisect(opt_func, 0, 5)

        # TODO
        raise Exception('Custom child process not implemented yet')
    
    else:
        raise ValueError(f'Invalid child_process provided of type {type(child_process)}')

    # If we don't want to use a buffer region, just set the value to zero.
    # Technically you could optimize further by actually not creating a copy
    # region, but it's not that big of a deal.
    if not use_parent_buffer:
        buffer_size = 0

    parent_region_rect = np.copy(region_rect)
    parent_region_rect[:,0] -= buffer_size
    parent_region_rect[:,1] += buffer_size

    # We'll create a new polygon, mostly for debug purposes so far.
    parent_region_boundary = np.array([[parent_region_rect[0,0], parent_region_rect[1,0]],
                                       [parent_region_rect[0,0], parent_region_rect[1,1]],
                                       [parent_region_rect[0,1], parent_region_rect[1,1]],
                                       [parent_region_rect[0,1], parent_region_rect[1,0]]])

    parent_region_rect_poly = Polygon(parent_region_boundary)

    # We should scale the number of parent points based on how much we increased
    # the area for the buffer.
    region_area = np.prod((region_rect[:,1] - region_rect[:,0]))
    parent_region_area = np.prod((parent_region_rect[:,1] - parent_region_rect[:,0]))
    scaled_parent_mean = parent_mean * parent_region_area / region_area

    # Generate the number of parent points
    num_parent_points = np.random.poisson(scaled_parent_mean)

    # Generate the parent points
    if type(parent_process) is str and parent_process.lower() == 'poisson':
        parent_points = generate_poisson(N=num_parent_points, d=d, region=parent_region_rect, diameter=None, density=parent_density)

    else:
        raise ValueError(f'Invalid parent process: {parent_process}.')

    # Generate the number of children that each parent will have.
    num_children = np.random.poisson(child_mean, size=(len(parent_points)))
    # We build this list iteratively.
    points = []
    parent_indices = []

    # There are a few different options for the generation order:
    # 1. We can do totally randomly, where we randomly select a child from
    # a random parent.
    # 2. We can do parent by parent, fully generating the children for a
    # parent before moving onto the next one.
    # 3. We can do child by child, generating one child for each parent
    # until there are none left to generate.
    #generation_order = 'parent'
    #generation_order = 'child'
    #generation_order = 'random'

    # Make a list of all the child points we need to generate.
    points_to_generate = np.array([j for i in range(len(num_children)) for j in np.repeat(i, num_children[i])], dtype=np.int64)

    # By default, this list is ordered parent by parent.
    if generation_order == 'parent':
        pass

    elif generation_order == 'random':
        # For random order, we just shuffle the indices
        np.random.shuffle(points_to_generate)

    elif generation_order == 'child':
        # Child is a bit harder
        # TODO
        raise ValueError(f'Child by child not implemented yet!')

    else:
        raise ValueError(f'Invalid generation order provided: {generation_order}')

    # Max 100 tries to generate the traps
    # This is chosen empirically, and the specific value isn't important.
    MAX_TRIES = 100

    for i in range(len(points_to_generate)):
        parent = parent_points[points_to_generate[i]]

        for j in range(MAX_TRIES):

            #distance = child_process_func()
            #angle = np.random.uniform(0, np.pi)
            #offset = np.array([np.cos(angle), np.sin(angle)]) * distance

            offset = child_process_func(size=2)

            new_child = parent + offset

            # Check the exclusion condition if necessary
            if (diameter is not None or soft_diameter is not None) and len(points) > 0:

                # We'll use a kdtree to make sure that the points aren't too close to
                # each other.
                kdtree = KDTree(points)

                # Find all points within the exclusion distance for each point.
                nearby_indices = kdtree.query_ball_point(new_child, diameter)
              
                # If there is another nearby point, we try again.
                if len(nearby_indices) > 0 and diameter is not None:
                    continue

                if soft_diameter is not None:
                    # Find the closest point
                    dist, ind = kdtree.query(new_child, 1)
                    
                    probability_reduction = np.exp(-dist/soft_diameter)

                    # If we draw a uniform random number below that probability
                    # reduction, reroll the new point.
                    if np.random.uniform(0, 1) < probability_reduction:
                        continue

            # If we properly found a new point, add it to the list and break
            # out of the loop over MAX_TRIES
            points += [new_child]
            parent_indices += [points_to_generate[i]]
            break

    points = np.array(points)
    parent_indices = np.array(parent_indices)

    # Remove points that are outside of the region.
    good_indices = [region_poly.contains(Point(p)) for p in points]
    points = points[good_indices]
    parent_indices = parent_indices[good_indices]

    # If we have zero points, we should just reroll
    if len(points) == 0:
        return generate_cluster(parent_mean, parent_process, parent_density, child_mean, child_process, child_process_scale, region, diameter, soft_diameter, use_parent_buffer, generation_order, debug)

    if debug:
        plt.plot(*region_poly.exterior.xy, c='tab:orange', linewidth=1, alpha=0.7, label='True region')
        #plt.plot(*region_rect_poly.exterior.xy, c='tab:blue', linewidth=1, alpha=0.7, label='Rectangle region')
        plt.plot(*parent_region_rect_poly.exterior.xy, c='tab:blue', linewidth=1, alpha=0.7, label='Parent region')

        plt.scatter(*parent_points.T, s=500, color='tab:blue', marker='+', label='Parents')

        plt.scatter(*points.T, s=30, color='tab:orange', marker='o', label='Children')
        # Draw lines from children to parents
        for i in range(len(points)):
            pi = parent_indices[i]
            plt.plot([parent_points[pi,0], points[i,0]], [parent_points[pi,1], points[i,1]], '--', color='black', linewidth=0.5)

        plt.legend(fontsize=12)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.show()

    return points


def generate_regular(N,
                     region=(1, 1), regularity=0, trapSize=.01):

    # Make sure it seems possible to fit this many points in the area
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

