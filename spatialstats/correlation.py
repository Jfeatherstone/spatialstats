import numpy as np

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.affinity import translate

from scipy.interpolate import RegularGridInterpolator
import time

import warnings
import itertools

import multiprocessing
import dill

# This overwrites multiprocessing's internal pickling with dill, which
# allows us to use local functions in parallelization.
dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler
multiprocessing.reduction.dump = dill.dump

def morisita_index(points,
                   region=None,
                   max_divisions=20,
                   use_diameter=False):
    """
    Compute the Morisita index, indicating the ratio of point pairs that
    fall into the same quadrat of the region versus different quadrats.

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

    use_diameter : bool
        Whether to return the number of quadrat divisions along one axis
        (default, False) or the diameter of the quadrats (True) alongside
        the index.

    Returns
    -------
    x_arr : numpy.ndarray[M]
        Array of the independent variable for the calculation, either the
        number of quadrats or the diameter of quadrats, depending on the
        value of ``use_diameter``.

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
    divisions_arr = np.arange(2, max_divisions+2)
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

    if use_diameter:
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

        I = \frac{m}{n (m - 1)} \left[ (n - 1) M - (n/m - 1) \right]

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
    divisions_arr = np.arange(2, max_divisions+2)
    index_arr = np.zeros(len(divisions_arr))

    # Transform the points such that one corner of the space is at the origin.
    adj_points = points - bounds[:,0]
    region_size = bounds[:,1] - bounds[:,0]

    total_ordered_pairs = len(points) * (len(points) - 1)

    for i, m in enumerate(divisions_arr):
        # Add some small padding to make sure points get included.
        padding = 1e-5
        quadrat_size = region_size / m + padding

        quadrat_identities = np.floor((adj_points + padding/2) / quadrat_size).astype(np.int64)

        # An easy way to calculate the number of pairs that share the same
        # quadrat is to compute the number of unique quadrat coordinates.
        # We don't care about which quadrat has how many points, so we can
        # just store the counts.
        _ , quadrat_point_counts = np.unique(quadrat_identities, return_counts=True, axis=0)

        # But then we also need to add all of the quadrats with zero
        # points in them
        #print(quadrat_point_counts)
        #print(m**d)

        quadrat_point_counts = np.concatenate((quadrat_point_counts, np.zeros(m**d - len(quadrat_point_counts))))

        # This is the (sample) variance divided by the mean.
        # ddof tells numpy to calculate the sample variance, ie. using m**d - 1
        # in the denominator instead of m**d. 
        index_arr[i] = np.var(quadrat_point_counts, ddof=1) / np.mean(quadrat_point_counts) #(n / m**d)

    if return_diameter:
        diameter_arr = np.array([np.linalg.norm(region_size / m) for m in divisions_arr])
        return diameter_arr, index_arr

    return divisions_arr**d, index_arr


def pair_correlation(points, bins=20, boundary=None, boundary_holes=None, kernel='epanechnikov', c=0.1, normalize_kernel=True, divisor='d', density=None, inhom_norm=1, verbose=1, plot=False, processes=1):
    r"""
    Computes the 2D radial distribution function (pair correlation function) 
    :math:``g(r)`` for a set of points, corrected to take account of the boundary
    effects.

    ``points`` can either be a single set of points, or can be multiple
    disjoints sets whose data will be pooled to compute the pair correlation
    function.
   
    The implementation is based on that of Antoine Bérut
    (https://github.com/aberut/paircorrelation2d) but modified to be able
    to pool data from separate sets of points.


    Parameters
    ---------- 
    points : numpy.ndarray[N,2] or list(numpy.ndarray[N_i,2])
        The (x,y) coordinates for a set containing N points, or a list of
        (unconnected) sets of points, for which the radial distribution function
        is to be calculated through pooling the data.
                
    bins : numpy.ndarray[m+1] or int
        A monotonically increasing array of the m+1 bin edges defining the
        m bins of the radius for which g(r) is going to be computed.

        If an integer m is given, the minimum bin size will be zero and the maximum
        bin size will be selected automatically, and a total of m+1 bins will
        be generated.
        
    boundary : numpy.ndarray[L,2] or shapely.geometry.Polygon or 'auto', optional
        The information needed to define the boundary enclosing the area
        of interest. 

        Can be provided as an ordered list of points that define the boundary,
        a Shapely Polygon, the string 'auto'.

        If 'auto', the convex hull of all provided points will be used.

        If None (default) no edge correction will be applied.

        Points that are outside the area of interest are automatically excluded.

        The list of coordinates must be "valid" in the sense used by the
        shapely library: linking all the points in order should result in a
        simple polygon with no line intersecting each other.
        
        eg. Assuming one wants a square border of area 1
            [[0,0],[0,1],[1,1],[1,0]] is valid (square shape)
            [[0,0],[0,1],[1,0],[1,1]] is not valid (bow tie shape)
                
    boundary_holes : list of numpy.ndarray[L_i, 2], optional
        A list of the (x,y) coordinates of points forming "holes" in the
        area of interest (useful if one is using a geometry with obstacles 
        or exclusion zones where no particles can be found).
        It is possible to define several holes (as long as they do not 
        intersect each other).
        
        eg. Assuming the area of interest is a square and one wants to 
            to remove a smaller square at the center
            boundary = [[0,0],[0,1],[1,1],[1,0]]
            boundary_holes = [[[0.2,0.2],[0.2,0.8],[0.8,0.8],[0.8,0.2]]]
        
        If no value is provided, the area of intereste will be a simple
        polygon with no hole.

    kernel : {'epanechnikov'} or None, optional
        The kernel to use in computing the pair correlation function.

        ``'epanechnikov'`` will use:

            k(x,w) = 3/(4w) max(1 - x^2 / w^2, 0)

        ``None`` will use no kernel, ie. just a value of 1 if the distance
        falls within the bin, and 0 otherwise.

    c : float, optional
        The Epanechnikov half width parameter. Recommended to have values
        between 0.1 and 0.2.

    normalize_kernel : bool, optional
        Whether or not to normalize the kernel by the accessible domain in
        computing contributions.

        Relevant for small radii, where if $r < h_c$, a portion of the kernel's
        density will not be accessible since it will requires $r < 0$. If
        True, normalize each contribution by integrating over the available
        density and dividing it by the total.

    divisor : {'d', 'r'}, optional
        What to divide the contributions to the g function by.

        ``'d'`` will use the actual distance of a given point, which usually
        gives better performance for small distance values (see Baddely et
        al., pg. 229).

        ``'r'`` will use the distance to the center of the bin.
       
    density : (numpy.ndarray[m,n,...], (numpy.ndarray[m], numpy.ndarray[n], ...)), optional
        The density field for the process if function is to be computed for
        inhomogeneous pattern. Should be given as a tuple including the multidimensional
        histogram values, and an array giving the bin centers for each axis.

    inhom_norm : int
        The power to which the inhomogeneous density correction is raised
        to. Typical values are 0 (if density field estimation is very
        accurate), 1, or 2.

    verbose : int, optional
        The level of verbosing when computing the pair correlation function.

        ``0`` means no messages or warnings.

        ``1`` means only warnings.

        ``2`` means warnings and timing information.

    plot : bool, optional
        If True, shows the points that were kept, and the computed g(r). 

    processes : int, optional
        The number of processes to use in calculating the normalization
        constants, which is typically the most time-intensive part of
        the calculation.

        If the value is a negative integer, that many cores will be left
        available on the system.


    Returns
    -------
    numpy.ndarray[m]
        The pair correlation function values for the m bins.

    numpy.ndarray[m]
        The centers of the radial distance bins corresponding to the values
        of the pair correlation function.

    """    

    # If you want to calculate the time taken, you can do that.
    if verbose > 1:
        t0 = time.time()
 
    # We first want to parse our points, since we might have multiple sets
    # or just a single one.
    try:
        # If this fails, we have a triangular array, and therefore more
        # than one set of points.
        points = np.array(points)

        # It could also be possible that we have multiple point sets all
        # with the same number of points, so we can check the number of
        # dims
        assert len(np.shape(points)) == 2

        point_sets = [np.array(points)]

    except:
        # Otherwise we already have a list of numpy arrays
        point_sets = points

    # For convenience, we will create a master list of all points, in order
    # to calculate a convex hull if needed.
    all_points = np.array([p for p_set in point_sets for p in p_set])

    # If we are given no boundary, we just compute the pair correlation
    # function with no correction
    if boundary is None:
        boundary_polygon = None

        # Even though we won't use a boundary, we still need the rough
        # area to be able to calculate the density.
        all_point_poly = MultiPoint(all_points)
        area = all_point_poly.convex_hull.area

    # If we are given the value of 'auto', we just use all points and take
    # the convex hull of them as our boundary.
    elif type(boundary) is str and boundary == 'auto':
        all_point_poly = MultiPoint(all_points)

        boundary_polygon = all_point_poly.convex_hull
        area = boundary_polygon.area

    # If we've been given a shapely polygon, we can use that
    elif type(boundary) is Polygon:
        boundary_polygon = boundary
        area = boundary_polygon.area

    # Otherwise, we should have been given an array[_like] that defines
    # the points on the boundary.
    else:
        if boundary_holes is None:
            boundary_polygon = Polygon(shell=boundary)
        else:
            boundary_polygon = Polygon(shell=boundary, holes=boundary_holes)
        
        if not boundary_polygon.is_valid:
            raise ValueError('The list of coordinates your provided for the border is not valid; see help for the definition of valid boundary and boundary_holes.')

        area = boundary_polygon.area
       
    # Now that we have our boundary defined properly, we will crop any points
    # that fall outside.
    if boundary_polygon is not None:
        cropped_point_sets = []
        for i in range(len(point_sets)):
            indices = np.where([boundary_polygon.contains(Point(p)) for p in point_sets[i]])[0]
            cropped_point_sets.append(point_sets[i][indices])
    else:
        cropped_point_sets = point_sets

    # If there are any that no longer have any points, give a warning.
    for i in range(len(cropped_point_sets)):
        if len(cropped_point_sets[i]) <= 1:
            if verbose > 0:
                warnings.warn(f'Set with index {i} has one or no points inside the specified boundary.')

    cropped_point_sets = [p_set for p_set in cropped_point_sets if len(p_set) > 1]

    # This is the mean density for the combined set of all points.
    mean_density = len(all_points) / area
    # And this is the mean density per set
    mean_density_per_set = np.array([len(p_set) for p_set in cropped_point_sets]) / area

    # Parse the bins value if need.

    # If we are given an integer, compute the max distance either using the
    # size of the boundary, or between any two points in the same set if
    # there is no boundary
    bin_padding = 4

    if not hasattr(bins, '__iter__') and int(bins) == bins:
        if boundary_polygon is None:
            set_bounds = [MultiPoint(cropped_point_sets[i]).bounds for i in range(len(cropped_point_sets))]
            set_sizes = [np.sqrt(np.sum((np.array(set_bounds[i][-2:]) - np.array(set_bounds[i][:2]))**2)) for i in range(len(cropped_point_sets))]

            max_radius = np.max(set_sizes) / 2

        # Otherwise, we can do the same for the boundary polygon
        else:
            max_radius = np.sqrt(np.sum((np.array(boundary_polygon.bounds[-2:]) - np.array(boundary_polygon.bounds[:2]))**2)) / 2

        # Ignore the first element so we don't have a zero
        # And add some extra indices so we can chop the last few at the
        # end. Otherwise the last bin will always be undersampled.
        radii = np.linspace(max_radius*0.01, max_radius * (1 + bin_padding/bins), bins + bin_padding)[:1]

    elif hasattr(bins, '__iter__'):
        # Add some extra padding bins on the end
        spacing = bins[1] - bins[0]
        radii = np.concatenate((bins, np.arange(1, bin_padding+1)*spacing + bins[-1]))
        # Can't have zero value
        radii = radii[radii > 0]

    else:
        raise ValueError(f'Invalid bins provided ({bins}); should be an array or integer.')

    # This will be used later on.
    bin_width = radii[1] - radii[0]

    # Set up our density interpolator if needed
    if hasattr(density, '__iter__'):
        inhomogeneous = True
        density_func = RegularGridInterpolator(density[1], density[0])

    else:
        inhomogeneous = False

    # Parse the kernel function
    if kernel is not None:
        if kernel.lower() == 'epanechnikov':
            # This uses the average density of each individual point pattern,
            # not the total density. I think this makes sense because using
            # the total density would give a very small kernel since we
            # are expect so many more points, and would result in a very
            # rough curve.
            def kernel_func(x, density):
                kernel_width = np.sqrt(5) * c / np.sqrt(density)
                return 3 / (4 * kernel_width) * max(0, 1 - x**2 / kernel_width**2)

            kernel_func = np.vectorize(kernel_func)
            
            # Compute the normalization correction if our kernel width is
            # greater than the half bin size and we start with our first
            # bin edge at 0. The idea is that in this case, that first bin
            # will be have a smaller domain of contributions from the kernel
            # function since part of its domain will be negative. I think it
            # makes sense to correct for the amount of the domain you are
            # missing.
            # NOTE: Probably very important to mention that I have never
            # seen this in any literature, so maybe this approach is totally
            # incorrect, though it seems reasonable to me.

            kernel_normalization_arr = np.ones((len(cropped_point_sets), len(radii)))

            if normalize_kernel:
                # If the first bin is zero and the kernel is larger than half the
                # bin size, we should correct for the amount of the domain that
                # isn't accessible to the first bin because it extends into the
                # negative axis.
                for i in range(len(cropped_point_sets)):
                    for j in range(len(radii)):
                        # Integrate the kernel over its entire domain
                        # We use the per set point density for this normalization
                        kernel_width = np.sqrt(5) * c / np.sqrt(mean_density_per_set[i])
                        kernel_normalization_const = np.sum(kernel_func(np.linspace(-kernel_width, kernel_width, 100), mean_density_per_set[i])) * (2 * kernel_width / 100)

                        if radii[j] - kernel_width < 0:
                            missing_portion = kernel_width - radii[j]
                            bin_normalization_const = np.sum(kernel_func(np.linspace(-kernel_width + missing_portion, kernel_width, 100), mean_density_per_set[i])) * ((2 * kernel_width - missing_portion) / 100)

                            kernel_normalization_arr[i,j] = bin_normalization_const / kernel_normalization_const

                        else:
                            break

        else:

            raise ValueError(f'Invalid kernel function passed: {kernel}')

    if verbose > 1:
        t1 = time.time() - t0
        print(f'Creating boundary polygon and sets of points inside took {t1:.3}s')
   
    # Next, we construct the rings of specified radii according to each size
    # we need for the binning for the normalization.

    # This function prints its own status messages, so no need to print
    # one after. See the function body for more information.
    if boundary_polygon is not None:
        normalization = _compute_ring_normalizations(boundary_polygon, cropped_point_sets, radii, processes, verbose)

    else:
        normalization = [np.ones((len(p_set), len(radii))) for p_set in cropped_point_sets]

    if verbose > 1:
        t3 = time.time() - t0

    # Compute the density (inhomogeneous) correction factors if necessary
    # The default value is just 1/lambda where lambda is the mean
    # density *for the total set of points*.

    density_arr = [np.ones(len(p_set)) * mean_density for p_set in cropped_point_sets]

    if inhomogeneous:
        for i in range(len(cropped_point_sets)):
            density_arr[i] = density_func(cropped_point_sets[i])

    # Next, we actually compute the pair correlation function.

    g_arr_normalized = np.zeros(len(radii))

    # For each point within a set, compute the distance to others in that
    # same set, then compute g(r) by binning
    for i in range(len(cropped_point_sets)):
        for j in range(len(cropped_point_sets[i])):
            distances = np.sqrt(np.sum((cropped_point_sets[i][j] - cropped_point_sets[i])**2, axis=-1))

            density_correction = density_arr[i][j] * density_arr[i]

            # Remove the self distance
            good_indices = np.where(distances > 0)

            distances = distances[good_indices]
            density_correction = density_correction[good_indices]

            if kernel is None:
                # Pretty arbitrary, but you should use a kernel.
                bin_edges = np.concatenate(([radii[0] - bin_width],
                                            (radii[1:] + radii[:-1]) / 2))

                if divisor == 'd':
                    # Compute the histogram
                    hist = np.histogram(distances, bins=bin_edges, weights=1/distances)[0]

                    # Normalize, and add to the running array
                    # The normalization constant involves the ring areas (which is
                    # needed to be consistent as a function of r) and the boundary term
                    g_arr_normalized += hist / (2 * np.pi * normalization[i][j])

                elif divisor == 'r':
                    # Compute the histogram
                    hist = np.histogram(distances, bins=bin_edges)[0]

                    # Normalize, and add to the running array
                    # The normalization constant involves the ring areas (which is
                    # needed to be consistent as a function of r) and the boundary term
                    g_arr_normalized += hist / (2 * np.pi * radii * normalization[i][j])

            else:
                # If we have a kernel, we need to go through each bin
                for k in range(len(radii)):
                    separations = distances - radii[k]

                    g_val_contributions = kernel_func(separations, mean_density_per_set[i])

                    distance_bins = np.floor(distances / bin_width).astype(np.int64)

                    good_indices = np.where(distance_bins < len(radii))

                    g_val_contributions = g_val_contributions[good_indices] / (density_correction[good_indices] * normalization[i][j,distance_bins[good_indices]])

                    # Convert to a fraction of total possible point pairs.
                    g_val_contributions /= (len(cropped_point_sets[i]) * (len(cropped_point_sets[i]) - 1))

                    if divisor == 'r':
                        g_arr_normalized[k] += np.nansum(g_val_contributions) / (2 * np.pi * radii[k] * kernel_normalization_arr[i,k])

                    elif divisor == 'd':
                        g_arr_normalized[k] += np.nansum(g_val_contributions / (2 * np.pi * distances[good_indices])) / kernel_normalization_arr[i,k]
            
    if verbose > 1:
        t4 = time.time() - t0
        print(f'Computing g(r) took {t4 - t3:.3} s')

    if inhomogeneous:
        # This value should be very close to 1, and represents how well
        # we have estimated our density field.
        inhom_normalization = np.sum([np.sum(1/arr) for arr in density_arr]) / area

        # Multiply by 1/D^p
        g_arr_normalized /= inhom_normalization**inhom_norm

    # Multiply by |A| lambda^2 / L
    # We have to divide by the number of point sets since we have
    # been adding values in the for loop above, so this gives an
    # average. The lambda^2 comes from the density correction, and
    # the area is just in the definition.
    g_arr_normalized *= mean_density**2 * area / len(cropped_point_sets)

    # This combination of dividing by the number density, but also dividing
    # by the area and number of points in each set above means a Poisson
    # distribution will have a value of 1.

    # Remove the extra bins at the end
    g_arr_normalized = g_arr_normalized[:-(bin_padding-1)]
    radii = radii[:-(bin_padding-1)]
    
    if plot:
        for i in range(len(cropped_point_sets)):
            plt.scatter(*cropped_point_sets[i].T)

        if boundary_polygon is not None:
            plt.plot(*boundary_polygon.exterior.xy, c='black', linewidth=3, alpha=0.7)

        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Point sets used to compute $g(r)$')
        plt.show()
        
        plt.plot(radii, g_arr_normalized)

        plt.xlabel('$r$')
        plt.ylabel('$g(r)$')
        plt.title('Radial distribution function')
        plt.show()
    
    if verbose > 1:
        t5 = time.time() - t0
        print(f'Total time: {t5}s for {len(all_points)} points ')
    
    return g_arr_normalized, radii


def ripley_k(points, bins=20, boundary=None, boundary_holes=None, linearize=False, density=None, inhom_norm=1, verbose=1, plot=False, processes=1):
    r"""
    Computes the 2D Ripley's K function (integral of the pair correlation function) 
    K(r) for a set of points, corrected to take account of the boundary effects.
   
    Note that this function implements the entire procedure of computing the
    K function, as opposed to simply integrating the pair correlation function.
    The reasoning was to cross-check and validate the pair correlation implementation
    above.

    Parameters
    ---------- 
    points : numpy.ndarray[N,2] or list(numpy.ndarray[N_i,2])
        The (x,y) coordinates for a set containing N points, or a list of
        (unconnected) sets of points, for which the radial distribution function
        is to be calculated through pooling the data.
                
    bins : numpy.ndarray[m+1] or int
        A monotonically increasing array of the m+1 bin edges defining the
        m bins of the radius for which g(r) is going to be computed.

        If an integer m is given, the minimum bin size will be zero and the maximum
        bin size will be selected automatically, and a total of m+1 bins will
        be generated.
        
    boundary : numpy.ndarray[L,2] or shapely.geometry.Polygon or 'auto', optional
        The information needed to define the boundary enclosing the area
        of interest. 

        Can be provided as an ordered list of points that define the boundary,
        a Shapely Polygon, the string 'auto'.

        If 'auto', the convex hull of all provided points will be used.

        If None (default) no edge correction will be applied.

        Points that are outside the area of interest are automatically excluded.

        The list of coordinates must be "valid" in the sense used by the
        shapely library: linking all the points in order should result in a
        simple polygon with no line intersecting each other.
        
        eg. Assuming one wants a square border of area 1
            [[0,0],[0,1],[1,1],[1,0]] is valid (square shape)
            [[0,0],[0,1],[1,0],[1,1]] is not valid (bow tie shape)
                
    boundary_holes : list of numpy.ndarray[L_i, 2], optional
        A list of the (x,y) coordinates of points forming "holes" in the
        area of interest (useful if one is using a geometry with obstacles 
        or exclusion zones where no particles can be found).
        It is possible to define several holes (as long as they do not 
        intersect each other).
        
        eg. Assuming the area of interest is a square and one wants to 
            to remove a smaller square at the center
            boundary = [[0,0],[0,1],[1,1],[1,0]]
            boundary_holes = [[[0.2,0.2],[0.2,0.8],[0.8,0.8],[0.8,0.2]]]
        
        If no value is provided, the area of intereste will be a simple
        polygon with no hole.

    linearize : bool, optional
        If True, the returned K values will transformed as $\\sqrt{K / \\pi}$.

        This makes it helpful to compare with Poisson data, which will appear
        as a straight line with slope 1, ie. y=x, and stabilizes the variance.

    density : (numpy.ndarray[m,n,...], (numpy.ndarray[m], numpy.ndarray[n], ...)), optional
        The density field for the process if function is to be computed for
        inhomogeneous pattern. Should be given as a tuple including the multidimensional
        histogram values, and an array giving the bin centers for each axis.

    inhom_norm : int
        The power to which the inhomogeneous density correction is raised
        to. Typical values are 0 (if density field estimation is very
        accurate), 1, or 2.
        
    verbose : int, optional
        The level of verbosing when computing the k function.

        ``0`` means no messages or warnings.

        ``1`` means only warnings.

        ``2`` means warnings and timing information.

    plot : bool, optional
        If True, shows the points that were kept, and the computed g(r). 
    

    Returns
    -------
    numpy.ndarray[m]
        The pair correlation function values for the m bins.

    numpy.ndarray[m]
        The centers of the radial distance bins corresponding to the values
        of the pair correlation function.

    """
    with warnings.catch_warnings():
        if verbose == 0:
            warnings.simplefilter("ignore")

        # All of the setup for this method is always exactly the same as for
        # the pair correlation function above.
        # If you want to calculate the time taken, you can do that.
        if verbose > 1:
            t0 = time.time()
     
        # We first want to parse our points, since we might have multiple sets
        # or just a single one.
        try:
            # If this fails, we have a triangular array, and therefore more
            # than one set of points.
            points = np.array(points)

            # It could also be possible that we have multiple point sets all
            # with the same number of points, so we can check the number of
            # dims
            assert len(np.shape(points)) == 2

            point_sets = [np.array(points)]

        except:
            # Otherwise we already have a list of numpy arrays
            point_sets = points

        # For convenience, we will create a master list of all points, in order
        # to calculate a convex hull if needed.
        all_points = np.array([p for p_set in point_sets for p in p_set])

        # If we are given no boundary, we just compute the k function with no correction.
        if boundary is None:
            # TODO check if this actually works
            boundary_polygon = None

        # If we are given the value of 'auto', we just use all points and take
        # the convex hull of them as our boundary.
        elif type(boundary) is str and boundary == 'auto':
            all_point_poly = MultiPoint(all_points)

            boundary_polygon = all_point_poly.convex_hull

        # If we've been given a shapely polygon, we can use that
        elif type(boundary) is Polygon:
            boundary_polygon = boundary

        # Otherwise, we should have been given an array[_like] that defines
        # the points on the boundary.
        else:
            if boundary_holes is None:
                boundary_polygon = Polygon(shell=boundary)
            else:
                boundary_polygon = Polygon(shell=boundary, holes=boundary_holes)
            
            if not boundary_polygon.is_valid:
                raise ValueError('The list of coordinates your provided for the border is not valid; see help for the definition of valid boundary and boundary_holes.')
           
        # Now that we have our boundary defined properly, we will crop any points
        # that fall outside.
        if boundary_polygon is not None:
            cropped_point_sets = []
            for i in range(len(point_sets)):
                indices = np.where([boundary_polygon.contains(Point(p)) for p in point_sets[i]])[0]
                cropped_point_sets.append(point_sets[i][indices])
        else:
            cropped_point_sets = point_sets

        # If there are any that no longer have 0 or 1 point, remove them
        # and give a warning.
        for i in range(len(cropped_point_sets)):
            if len(cropped_point_sets[i]) == 0:
                warnings.warn(f'Set with index {i} has no points inside the specified boundary.')

        cropped_point_sets = [p for p in cropped_point_sets if len(p) > 1]

        # Set up our density interpolator if needed
        if hasattr(density, '__iter__'):
            inhomogeneous = True
            density_func = RegularGridInterpolator(density[1], density[0])

        else:
            inhomogeneous = False

        # Final setup thing: parse the bins value if need.

        # If we are given an integer, compute the max distance either using the
        # size of the boundary, or between any two points in the same set if
        # there is no boundary
        if not hasattr(bins, '__iter__') and int(bins) == bins:
            if boundary_polygon is None:
                set_bounds = [MultiPoint(cropped_point_sets[i]).bounds for i in range(len(cropped_point_sets))]
                set_sizes = [np.sqrt(np.sum((np.array(set_bounds[i][-2:]) - np.array(set_bounds[i][:2]))**2)) for i in range(len(cropped_point_sets))]

                max_radius = np.max(set_sizes) / 2

            # Otherwise, we can do the same for the boundary polygon
            else:
                max_radius = np.sqrt(np.sum((np.array(boundary_polygon.bounds[-2:]) - np.array(boundary_polygon.bounds[:2]))**2)) / 2

            bin_edges = np.linspace(0, max_radius, bins)

        elif hasattr(bins, '__iter__'):
            bin_edges = bins

        else:
            raise ValueError(f'Invalid bins provided ({bins}); should be an array or integer.')

        non_zero_bin_edges = np.array([b for b in bin_edges if b > 0])

        bin_width = non_zero_bin_edges[1] - non_zero_bin_edges[0]

        if verbose > 1:
            t1 = time.time() - t0
            print(f'Creating boundary polygon and sets of points inside took {t1:.3}s')
       
        # Next, we construct the rings of specified radii according to each size
        # we need for the binning for the normalization.

        # This function prints its own status messages, so no need to print
        # one after. See the function body for more information.
        normalization = _compute_ring_normalizations(boundary_polygon, cropped_point_sets, non_zero_bin_edges, processes, verbose)

        if verbose > 1:
            t3 = time.time() - t0
       
        # Compute the density (inhomogeneous) correction factors if necessary
        # The default value is just 1/lambda where lambda is the mean
        # density *for the total set of points*.
        mean_density = len(all_points) / boundary_polygon.area

        density_arr = [np.ones(len(p_set)) * mean_density for p_set in cropped_point_sets]

        if inhomogeneous:
            for i in range(len(cropped_point_sets)):
                density_arr[i] = density_func(cropped_point_sets[i])

        # Next, we actually compute the k function.
        k_arr_normalized = np.zeros(len(non_zero_bin_edges))
        
        # For each point within a set, compute the distance to others in that
        # same set, then compute g(r) by binning
        for i in range(len(cropped_point_sets)):
            for j in range(len(cropped_point_sets[i])):
                distances = np.sqrt(np.sum((cropped_point_sets[i][j] - cropped_point_sets[i])**2, axis=-1))

                density_correction = density_arr[i][j] * density_arr[i]

                # Remove the self distance
                good_indices = np.where(distances > 0)

                distances = distances[good_indices]
                density_correction = density_correction[good_indices]

                # Identify the number of distances less than each bin size
                # Don't need to sort unless we're using searchsorted.
                #sort_order = np.argsort(distances)

                #distances = distances[sort_order]
                #density_correction = density_correction[sort_order]

                # Not using k as an index because that would be confusing :)
                for l in range(len(non_zero_bin_edges)):
                    # This method is faster, but isn't compatible with inhomogeneous
                    # point patterns.
                    # We use side='right' because we want to count every distance
                    # that is less than *or equal* to the circle radius.
                    #k_val = np.searchsorted(distances, bin_edges[l], side='right')

                    # Instead, we have to look at all the specific locations
                    # of distances less than the current size, and add up
                    # the inverse of they density corrections. If we are
                    # in the homogeneous case, inhom_correction is all ones,
                    # so this is equivalent to above.
                    neighbor_indices = np.where(distances <= non_zero_bin_edges[l])
                    distance_bins = np.floor(distances[neighbor_indices] / bin_width).astype(np.int64)

                    # Note that despite the fact that we are iterating over
                    # the bins with l, not every neighbor point will use
                    # the same correction, but could use any with index
                    # from 0 to l.
                    k_val_contributions = 1 / density_correction[neighbor_indices] / normalization[i][j,distance_bins]
                   
                    # Divide by the edge normalization, and n (n - 1) for this
                    # set of points, which converts the k value into a fraction
                    # of the total number of possible pairs (accounting for this
                    # specific point set).
                    k_arr_normalized[l] += np.sum(k_val_contributions) / (len(cropped_point_sets[i]) * (len(cropped_point_sets[i]) - 1))

        if verbose > 1:
            t4 = time.time() - t0
            print(f'Computing k(r) took {t4 - t3:.3} s')
  
        if inhomogeneous:
            # This value should be very close to 1, and represents how well
            # we have estimated our density field.
            inhom_normalization = np.sum([np.sum(1/arr) for arr in density_arr]) / boundary_polygon.area

            # Multiply by 1/D^p
            k_arr_normalized /= inhom_normalization**inhom_norm

        # Multiply by |A| lambda^2 / L
        # We have to divide by the number of point sets since we have
        # been adding values in the for loop above, so this gives an
        # average. The lambda^2 comes from the density correction, and
        # the area is just in the definition.
        k_arr_normalized *= mean_density**2 * boundary_polygon.area / len(cropped_point_sets)

        if linearize:
            k_arr_normalized = np.sqrt(k_arr_normalized / np.pi)

        if plot:
            for i in range(len(cropped_point_sets)):
                plt.scatter(*cropped_point_sets[i].T)

            if boundary_polygon is not None:
                plt.plot(*boundary_polygon.exterior.xy, c='black', linewidth=3, alpha=0.7)

            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.title('Point sets used to compute $K(r)$')
            plt.show()
            
            plt.plot(non_zero_bin_edges, k_arr_normalized)

            plt.xlabel('$r$')
            plt.title('Ripley\'s $K$')
            if linearize: 
                plt.ylabel('Ripley\'s $\\hat L = \\sqrt{\\hat K / \\pi}$')
            else:
                plt.ylabel('Ripley\'s $\hat K$')

            plt.show()
        
        if verbose > 1:
            t5 = time.time() - t0
            print(f'Total time: {t5}s for {len(all_points)} points ')
    
    return k_arr_normalized, non_zero_bin_edges


def _compute_ring_normalizations(region, point_sets, radii, processes=1, verbose=1):
    """
    """
    if verbose > 1:
        t0 = time.time()

    # The proper edge correction uses the fraction of the circumference that
    # falls within the region as the normalization factor. For more information,
    # see the book:
    # Spatial Point Patterns: Methodology and Applications in R, section 7.4.3

    # Actually computing the fraction of the circumference is a little bit difficult
    # and it is much easier to compute the area that intersects a region.
    # So we create relatively thin rings covering radii [r - eps, r + eps], compute
    # the area they overlap with the region, and then divide by the width
    # of those rings  (2*eps) to get an approximation of the area.
    # This approximation is pretty good, and I couldn't find any particular
    # case where the error was greater than ~0.2% (except when eps > r, though
    # that isn't really relevant).

    rings = [[]]*len(radii)
    # This is not the area of the thin ring we create, but the area of the
    # proper disk with radius r.
    disk_areas = np.zeros(len(radii))

    # TODO: What should I choose for this?
    ring_half_width = (radii[1] - radii[0]) / 2

    for i in range(len(radii)):
        inner_circle = Point([0,0]).buffer(radii[i] - ring_half_width)
        outer_circle = Point([0,0]).buffer(radii[i] + ring_half_width)

        rings[i] = outer_circle.difference(inner_circle)
       
        disk_areas[i] = np.pi * ((radii[i] + ring_half_width)**2 - (radii[i] - ring_half_width)**2)

    if verbose > 1:
        t1 = time.time() - t0
        print(f'Creating all ring polygons took {t1:.3}s')
   
    # Next, we are going to compute the normalization factors for each
    # point and each bin.

    #For each point, computes its distance to the boundary, and for each bin computes the normalization factor
    #(the area of "the intersection of the ring and the area of interest", divided by the ring area)
    normalization = [np.ones((len(p_set), len(radii))) for p_set in point_sets]

    # If we use just a single process, we just run through the nested
    # for loops naively.
    if processes is None or processes == 1:
      
        # Now we will translate each of those rings to each point and compute
        # how the ring intersects with the boundary.
        for i in range(len(point_sets)):
            for j in range(len(point_sets[i])):
                for k in range(len(radii)):
                    if radii[k] == 0:
                        normalization[i][j,k] = np.nan

                    # The time taken for these operations is about 40% for
                    # the translation, and 60% for the intersection.
                    translated_ring = translate(rings[k], xoff=point_sets[i][j,0], yoff=point_sets[i][j,1])
                    # First divide the area by (2 dr) to approximate the circumference,
                    # then divide by the true circumference to get the fraction.
                    normalization[i][j,k] = region.intersection(translated_ring).area / (2 * ring_half_width) / (2 * np.pi * radii[k])

    elif processes > 1 or processes < 0:

        # Parse the number of processes
        num_processes = processes if processes > 0 else multiprocessing.cpu_count() + processes
        assert num_processes > 0, f'Invalid number of processes specified ({processes}); only {multiprocessing.cpu_count()} are available.'

        lock = multiprocessing.Lock()

        def process_func(i):
            norm_values = np.zeros((len(point_sets[i]), len(radii)))
            # We still loop over the bins in each process, since otherwise
            # the time spent on creating and stopping processes would
            # far outweigh the benefit of parallelizing.
            for j in range(len(norm_values)):
                for k in range(len(radii)):
                    if radii[k] == 0:
                        normalization[i][j,k] = np.nan

                    translated_ring = translate(rings[k], xoff=point_sets[i][j,0], yoff=point_sets[i][j,1])
                    # First divide the area by (2 dr) to approximate the circumference,
                    # then divide by the true circumference to get the fraction.
                    norm_values[j,k] = region.intersection(translated_ring).area / (2 * ring_half_width) / (2 * np.pi * radii[k])

            return i, norm_values

        # If we are parallelizing, we first need to generate the list of
        # jobs, and then we can assign them to a pool.
        job_list = np.arange(len(point_sets))
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(process_func, job_list):
                i, val = result
                lock.acquire()
                normalization[i] = val
                lock.release()

    if verbose > 1:
        t2 = time.time() - t0
        print(f'Computing normalization factors took {t2 - t1:.3}s')

    # Give a warning if any of the normalization values are zero, as it means
    # the we are considering too large of a ring. We'll also set all of
    # those values to nan, since otherwise we would divide by zero.
    bad_indices = [np.where(normalization[i] == 0) for i in range(len(normalization))]

    total_bad_indices = np.sum([len(bad_indices[i][0]) for i in range(len(normalization))])

    if total_bad_indices > 0:
        if verbose > 0:
            warnings.warn('Some normalization constants are zero, meaning too large of a radius (bin size) was used!')

        for i in range(len(normalization)):
            normalization[i][bad_indices[i]] = np.nan

    return normalization
