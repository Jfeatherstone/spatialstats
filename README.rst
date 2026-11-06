Spatial point statistics
========================

.. image:: https://github.com/jfeatherstone/spatialstats/workflows/Tests/badge.svg
   :target: https://github.com/jfeatherstone/spatialstats/actions
   :alt: Tests

Two-dimensional spatial point statistics implemented in Python.

Analysis Usage
--------------

.. code-block:: python

    import spatialstats as ss

    # Some 2D point data
    data = np.random.uniform(0, 1, size=(100,2))

    # Compute the pair correlation function
    g_arr, g_bin_arr = ss.pair_correlation(data)

    # Compute the Ripley K function
    k_arr, k_bin_arr = ss.ripley_k(data)

    ...

Generation Usage
----------------

.. code-block:: python

    import spatialstats as ss
    from shapely.geometry import Polygon

    L = 1
    region = Polygon([[0, 0], [L, 0], [L, L], [0, L]])

    # Generate points from a Poisson process in our region
    points = ss.generate_poisson(100, region=region)

    # Generate points from a hard core Poisson process
    points = ss.generate_poisson(100, region=region, diameter=0.1)

    # Generate points from a Thomas cluster process
    points = ss.generate_cluster(parent_mean=5, child_mean=5, region=region)

    # Generate points from a soft core Thomas cluster process
    points = ss.generate_cluster(parent_mean=5, child_mean=5, region=region, soft_diameter=0.1)

    ...

Installation
------------
You can install directly from the source:

.. code-block:: console

    git clone https://github.com/jfeatherstone/spatialstats
    cd spatialstats
    pip install .

Development
-----------

To run the test suite, we recommend using pytest:

.. code-block:: console

    python -m pytest testing/ -v
