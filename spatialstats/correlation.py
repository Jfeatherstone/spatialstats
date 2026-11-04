import numpy as np

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.affinity import translate
import time

import warnings

import multiprocessing

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
    divisions_arr = np.arange(2, max_divisions)
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


def pair_correlation(points, bins=20, boundary=None, boundary_holes=None, debug=False, plot=False, processes=1):
    r"""
    Computes the 2D radial distribution function (pair correlation function) 
    g(r) for a set of points, corrected to take account of the boundary effects.
   
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
        
    debug : bool, optional
        If True, the code will print outputs showing timings at different
        stages of the computation (to let one know what operations are the
        most time consuming).

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
    if debug:
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
        if len(cropped_point_sets[i]) == 0:
            warnings.warn(f'Set with index {i} has no points inside the specified boundary.')

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

    # This will be used later on in the fast calculation and returned
    # at the end.
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    if debug:
        t1 = time.time() - t0
        print(f'Creating boundary polygon and sets of points inside took {t1:.3}s')
   
    # Next, we construct the rings of specified radii according to each size
    # we need for the binning for the normalization.
        
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

    rings = [[]]*(len(bin_edges)-1)
    # This is not the area of the thin ring we create, but the area of the
    # proper disk with radius r.
    disk_areas = np.zeros(len(bin_edges)-1)

    # TODO: What should I choose for this?
    ring_half_width = (bin_edges[1] - bin_edges[0]) / 2

    for i in range(len(bin_edges)-1):
        inner_circle = Point([0,0]).buffer(bin_centers[i] - ring_half_width)
        outer_circle = Point([0,0]).buffer(bin_centers[i] + ring_half_width)

        rings[i] = outer_circle.difference(inner_circle)
       
        disk_areas[i] = np.pi * ((bin_centers[i] + ring_half_width)**2 - (bin_centers[i] - ring_half_width)**2)

    if debug:
        t2 = time.time() - t0
        print(f'Creating all ring polygons took {t2 - t1:.3}s')
   
    # Next, we are going to compute the normalization factors for each
    # point and each bin.

    #For each point, computes its distance to the boundary, and for each bin computes the normalization factor
    #(the area of "the intersection of the ring and the area of interest", divided by the ring area)
    normalization = [np.ones((len(p_set), len(bin_edges) - 1)) for p_set in cropped_point_sets]

    
    # 
    #lock = multiprocessing.Lock()

    if boundary_polygon is not None:
        # I thought this fast method would actually help but it turns out that
        # it is much worse that the regular one haha. So I'll leave it in
        # case it ends up being useful for something, but definitely don't
        # use it by default.
        fast = False
        if fast:
            # For the fast method, we aren't going to compute proper intersections
            # of the rings with the boundary, but instead just see how many equally
            # spaced points around the boundary sit inside the boundary, and use
            # that as a proxy for the total length of circumference inside.
            discrete_steps = 30

            angle_steps = np.linspace(0, 2*np.pi, discrete_steps+1)[:-1]

            unit_circle_points = np.array([np.sin(angle_steps), np.cos(angle_steps)]).T

            for i in range(len(cropped_point_sets)):
                for j in range(len(cropped_point_sets[i])):
                    for k in range(len(bin_centers)):
                        test_points = unit_circle_points*bin_centers[k] + cropped_point_sets[i][j]
                        normalization[i][j,k] = len(np.where([boundary_polygon.contains(Point(p)) for p in test_points])[0]) / discrete_steps
            
        else:
            # For each distance bin, define the ring (difference between the disk
            # of radius r[i+1] and the disk of radius r[i]) and computes the area of
            # those rings.

            #rings = [None]*(len(bin_edges) - 1)
            #ring_areas = np.zeros(len(bin_edges) - 1)

            # For each distance bin, define the ring (difference between the disk
            # of radius r[i+1] and the disk of radius r[i]) and computes the area of
            # those rings.
            #for k in range(len(bin_edges)-1):
            #    inner_circle = Point([0,0]).buffer(bin_edges[k])
            #    outer_circle = Point([0,0]).buffer(bin_edges[k+1])

            #    rings[k] = outer_circle.difference(inner_circle)
            #    ring_areas[k] = rings[k].area
          
            # Now we will translate each of those rings to each point and compute
            # how the ring intersects with the boundary.
            # TODO: This is the part that takes the most time, maybe could speed
            # it up somehow?
            for i in range(len(cropped_point_sets)):
                for j in range(len(cropped_point_sets[i])):
                    for k in range(len(bin_edges) - 1):
                        # The time taken for these operations is about 40% for
                        # the translation, and 60% for the intersection.
                        translated_ring = translate(rings[k], xoff=cropped_point_sets[i][j,0], yoff=cropped_point_sets[i][j,1])
                        # First divide the area by (2 dr) to approximate the circumference,
                        # then divide by the true circumference to get the fraction.
                        normalization[i][j,k] = boundary_polygon.intersection(translated_ring).area / (2 * ring_half_width) / (2 * np.pi * bin_centers[k])

    if debug:
        t3 = time.time() - t0
        print(f'Computing normalization factors took {t3 - t2:.3}s')
    
    # Next, we actually compute the pair correlation function.

    g_arr_normalized = np.zeros(len(bin_edges) - 1)
    
    # For each point within a set, compute the distance to others in that
    # same set, then compute g(r) by binning
    for i in range(len(cropped_point_sets)):
        for j in range(len(cropped_point_sets[i])):
            distances = np.sqrt(np.sum((cropped_point_sets[i][j] - cropped_point_sets[i])**2, axis=-1))

            # Remove the self distance
            distances = distances[distances > 0]

            # Compute the histogram
            hist = np.histogram(distances, bins=bin_edges)[0]

            # Normalize, and add to the running array
            # The normalization constant involves the ring areas (which is
            # needed to be consistent as a function of r) and the boundary term
            g_arr_normalized += hist / (disk_areas * normalization[i][j])
            
    if debug:
        t4 = time.time() - t0
        print(f'Computing g(r) took {t4 - t3:.3} s')

    # Multiply by |A| / N (N-1)
    g_arr_normalized *= area / np.sum([len(p) * (len(p) - 1) for p in cropped_point_sets])

    # This combination of dividing by the number density, but also dividing
    # by the area and number of points in each set above means a Poisson
    # distribution will have a value of 1.
    
    if plot:
        for i in range(len(cropped_point_sets)):
            plt.scatter(*cropped_point_sets[i].T)

        if boundary_polygon is not None:
            plt.plot(*boundary_polygon.exterior.xy, c='black', linewidth=3, alpha=0.7)

        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Point sets used to compute $g(r)$')
        plt.show()
        
        plt.plot(bin_centers, g_arr_normalized)

        plt.xlabel('$r$')
        plt.ylabel('$g(r)$')
        plt.title('Radial distribution function')
        plt.show()
    
    if debug:
        t5 = time.time() - t0
        print(f'Total time: {t5}s for {len(all_points)} points ')
    
    return g_arr_normalized, bin_centers


def ripley_k(points, bins=20, boundary=None, boundary_holes=None, linearize=False, debug=False, plot=False):
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
        If True, the returned K values will be divided by \\pi r.

        This makes it helpful to compare with Poisson data, which will appear
        as a straight line with slope 1, ie. y=x.
        
    debug : bool, optional
        If True, the code will print outputs showing timings at different
        stages of the computation (to let one know what operations are the
        most time consuming).

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
    # All of the setup for this method is always exactly the same as for
    # the pair correlation function above.
    # If you want to calculate the time taken, you can do that.
    if debug:
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

    # If there are any that no longer have any points, give a warning.
    for i in range(len(cropped_point_sets)):
        if len(cropped_point_sets[i]) == 0:
            warnings.warn(f'Set with index {i} has no points inside the specified boundary.')

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

    if debug:
        t1 = time.time() - t0
        print(f'Creating boundary polygon and sets of points inside took {t1:.3}s')
   
    # Next, we construct the circles of specified radii according to each size
    # we need for the binning for the normalization.

    # Unlike in the pair_correlation function, we don't need to constantly
    # use len(bin_edges) - 1, since we aren't considering intervals of
    # space, but just radii.

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

    rings = [[]]*(len(bin_edges))
    # This is not the area of the thin ring we create, but the area of the
    # proper disk with radius r.
    disk_areas = np.zeros(len(bin_edges))

    # TODO: What should I choose for this?
    ring_half_width = (bin_edges[1] - bin_edges[0]) / 2

    for i in range(len(bin_edges)):
        inner_circle = Point([0,0]).buffer(bin_edges[i] - ring_half_width)
        outer_circle = Point([0,0]).buffer(bin_edges[i] + ring_half_width)

        rings[i] = outer_circle.difference(inner_circle)
       
        disk_areas[i] = np.pi * bin_edges[i]**2

    if debug:
        t2 = time.time() - t0
        print(f'Creating all circle polygons took {t2 - t1:.3}s')
   
    # Next, we are going to compute the normalization factors for each
    # point and each bin.
    normalization = [np.ones((len(p_set), len(bin_edges))) for p_set in cropped_point_sets]

    # Now we will translate each of those circles to each point and compute
    # how the circle intersects with the boundary.
    # TODO: This is the part that takes the most time, maybe could speed
    # it up somehow?
    for i in range(len(cropped_point_sets)):
        for j in range(len(cropped_point_sets[i])):
            for l in range(len(bin_edges)):
                if bin_edges[l] == 0:
                    continue

                # The time taken for these operations is about 40% for
                # the translation, and 60% for the intersection.
                translated_ring = translate(rings[l], xoff=cropped_point_sets[i][j,0], yoff=cropped_point_sets[i][j,1])
                # First divide the area by (2 dr) to approximate the circumference,
                # then divide by the true circumference to get the fraction.
                normalization[i][j,l] = boundary_polygon.intersection(translated_ring).area / (2 * ring_half_width) / (2 * np.pi * bin_edges[l])

    if debug:
        t3 = time.time() - t0
        print(f'Computing normalization factors took {t3 - t2:.3}s')
    
    # Next, we actually compute the k function.
    k_arr_normalized = np.zeros(len(bin_edges))
    
    # For each point within a set, compute the distance to others in that
    # same set, then compute g(r) by binning
    for i in range(len(cropped_point_sets)):
        for j in range(len(cropped_point_sets[i])):
            distances = np.sqrt(np.sum((cropped_point_sets[i][j] - cropped_point_sets[i])**2, axis=-1))

            # Remove the self distance
            distances = distances[distances > 0]

            # Identify the number of distances less than each bin size
            np.sort(distances)

            # Not using k as an index because that would be confusing :)
            for l in range(len(bin_edges)):
                if bin_edges[l] == 0:
                    continue

                # We use side='right' because we want to count every distance
                # that is less than *or equal* to the circle radius.
                k_val = np.searchsorted(distances, bin_edges[l], side='right')
               
                # Divide by the normalization
                k_arr_normalized[l] += k_val / (normalization[i][j,l])

    if debug:
        t4 = time.time() - t0
        print(f'Computing g(r) took {t4 - t3:.3} s')
  
    # Multiply by |A| / N (N-1)
    k_arr_normalized *= boundary_polygon.area / np.sum([len(p) * (len(p) - 1) for p in cropped_point_sets])

    if linearize:
        k_arr_normalized /= np.pi * bin_edges

    if plot:
        for i in range(len(cropped_point_sets)):
            plt.scatter(*cropped_point_sets[i].T)

        if boundary_polygon is not None:
            plt.plot(*boundary_polygon.exterior.xy, c='black', linewidth=3, alpha=0.7)

        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Point sets used to compute $K(r)$')
        plt.show()
        
        plt.plot(bin_edges, k_arr_normalized)

        plt.xlabel('$r$')
        plt.ylabel('$K(r)$')
        if linearize: 
            plt.title('Ripley\'s $K/\\pi r$')
        else:
            plt.title('Ripley\'s $K$')
        plt.show()
    
    if debug:
        t5 = time.time() - t0
        print(f'Total time: {t5}s for {len(all_points)} points ')
    
    return k_arr_normalized, bin_edges
