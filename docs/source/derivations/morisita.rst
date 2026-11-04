Indices of dispersion
---------------------

This page covers the derivation of the Morisita index and it's relationship
to the index of dispersion of quadrat counts.

Morisita index
==============

The Morisita index is defined [1,2] by breaking up the space into :math:`m` equally-sized
*quadrats*, and then taking the total number of pairs of points that fall in
the same quadrat divided by the number of pairs of points in the whole system:

.. math::

    M(m) = m \frac{\sum_j^m n_j (n_j - 1)}{n (n - 1)}.

:math:`n_j` is the number of points that fall into the :math:`j^{th}` quadrat.
You typically compute this quantity for many different values of :math:`m`,
and then plot the index as a function either directly of :math:`m`, or as a
function of the diameter of each quadrat, :math:`d(m)`.

When the points are independently distributed, the probability of any two pairs
being in the same quadrat will be :math:`1/m`, which leads to a Morisita index
of one. A value above one means points are more clustered together on average,
and below one means they are regularly-spaced.

Index of dispersion of quadrat counts
=====================================

Alternatively, we could look at the ratio of the variance to the mean for
the numbers of points in each quadrat, :math:`{n_j}`:

.. math::

    I(m) = \frac{\sigma^2}{\mu}

There is a similar interpretation for this index of dispersion: if we have an
independent (Poisson) distribution of points, the variance should be equal
to the mean, so this will have a value around one. Higher or lower than one
means that the points are either clustered or regularly-spaced, respectively.

Morisita's index and the index of dispersion of quadrat counts contain the
same information about the point distribution, and one can always be calculated
from the other, derived below.

.. math::

    I(m) =  \frac{1}{\mu} \left( \frac{1}{m - 1} \sum_j^m (n_j - \hat \mu)^2 \right)

Since we are interested in comparing to independent or Poisson distributions,
we can make the assumption that :math:`\mu = n / m`. We'll assume that the
sample mean :math:`\hat \mu` and the true mean :math:`\mu` are the same.

.. math::

    I(m) =  \frac{m}{n (m - 1)} \left(\sum_j^m (n_j - \frac{n}{m})^2 \right)

.. math::

    I(m) =  \frac{m}{n (m - 1)} \left(\sum_j^m \left(n_j^2 + (\frac{n}{m})^2 -  \frac{2 n n_j}{m})\right) \right)

.. math::

    I(m) =  \frac{m}{n (m - 1)} \left(\sum_j^m \left(n_j^2 + (\frac{n}{m})^2 -  \frac{2 n n_j}{m}) \right) \right)

.. math::

    I(m) =  \frac{m}{n (m - 1)} \left(\sum_j^m n_j^2 - \frac{n^2}{m} \right)

Rearranging the definition of the Morisita index, we get:

.. math::

    M(m) = \frac{m}{n (n - 1)}\left( \sum_j^m n_j^2 - n \right).

When combined with the expression for :math:`I(m)`, we have:

.. math::

    M(m) = \frac{1}{n - 1} \left[ I(m) (m - 1) + n - m \right].

References
==========

[1] Morisita M. (1961). Measuring of dispersion of individuals and analysis of the distributional patterns. (No. 6). 一般社団法人 日本生態学会. https://doi.org/10.18960/seitai.11.6_252_3

[2] Baddeley, A., Rubak, E., & Turner, R. (2015). Spatial Point Patterns: Methodology and Applications with R. Chapman and Hall/CRC. https://doi.org/10.1201/b19708

