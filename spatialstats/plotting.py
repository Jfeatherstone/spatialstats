import numpy as np
import matplotlib.pyplot as plt

from .correlation import morisita_index

def morisita_index_plot(points,
                        region=None,
                        max_divisions=20,
                        use_diameter=False,
                        ax=None,
                        **kwargs):
    """
    Plot the Morisita index, indicating the ratio of point pairs that
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
        Whether to plot the index as a function of the number of quadrat
        divisions along one axis (default, False) or the diameter of the
        quadrats (True).
    
    ax : matplotlib.axis, optional
        The axis to plot the index on. If not provided, a new axis will
        be created.

    kwargs : keyword arguments for matplotlib.pyplot.plot

    Returns
    -------
    fig : matplotlib.figure
        Figure upon which the index is plotted.
    """

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5,4))

    m_arr, mori_arr = morisita_index(points, region, max_divisions, use_diameter)

    # We only plot this horizontal line if there isn't already a line
    # with the lable 'Poisson' on the plot
    if 'Poisson' not in [l.get_label() for l in ax.lines]:
        ax.axhline(1, linestyle='--', c='gray', label='Poisson')

    # Pass kwargs to the plot function, and add markers in the user hasn't
    # specified
    if not 'fmt' in kwargs:
        ax.plot(m_arr, mori_arr, '-o', **kwargs)

    else:
        ax.plot(m_arr, mori_arr, **kwargs)
        
    ax.set_xscale('log')
        
    if use_diameter:
        ax.set_xlabel('Quadrat diameter, $d(m)$')

    else:
        ax.set_xlabel('Quadrat number, $m$')

    ax.set_ylabel('Morisita index, $M(m)$')

    ax.legend()

    return plt.gcf()


def fry_plot(points, ax=None, **kwargs):
    """
    Fry or Patterson plot of the data, showing difference vectors between
    all simultaneous points.

    Parameters
    ----------
    points : numpy.ndarray[N,2] or list(numpy.ndarray[N,2])
        The set of points to compute the plot for. If given as a list of
        sets of points, the difference vectors will be calculated only
        within each sample, but will all be plotted together.
    """

    try:
        # If this fails, we have a triangular array, and therefore more
        # than one set of points.
        points = np.array(points)
        point_sets = [np.array(points)]

    except:
        point_sets = points

    diff_vectors = []

    for i in range(len(point_sets)):
        for j in range(len(point_sets[i])):
            diff_vectors += list(point_sets[i][j] - point_sets[i])
   
    diff_vectors = np.array(diff_vectors)
    # Remove points at (0,0) (self difference vectors)
    diff_vectors[np.sum(diff_vectors**2, axis=-1) < 1e-10] = np.nan

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    ax.scatter(*diff_vectors.T, **kwargs)
    ax.scatter([0], [0], marker='+', color='black', s=100)

    ax.set_title('Fry Plot')

    
    return plt.gcf()

