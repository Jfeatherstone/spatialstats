powerlaw: A Python Package for Analysis of Heavy-Tailed Distributions
=====================================================================

.. image:: https://github.com/jfeatherstone/spatialstats/workflows/Tests/badge.svg
   :target: https://github.com/jfeatherstone/spatialstats/actions
   :alt: Tests

Spatial point statistics implemented in Python.

Basic Usage
------------
The most basic use of this library is to fit some data, extract parameters,
and make comparisons to other distributions:

Installation
------------
You can install directly from the source:

.. code-block:: console

    $ git clone https://github.com/jeffalstott/powerlaw
    $ cd powerlaw
    $ pip install .

Development
-----------

To run the test suite, we recommend using pytest:

.. code-block:: console

    python -m pytest testing/ -v
