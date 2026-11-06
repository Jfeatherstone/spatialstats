import numpy as np

from scipy.optimize import minimize

import matplotlib.pyplot as plt

from .generate import generate_cluster
from .correlation import pair_correlation

def fit_cluster_process(data, initial_guess={}, parent_process='poisson', child_process='gaussian', region=None, homogeneous=True, use_hard_diameter=False, use_soft_diameter=False, generation_order='parent', debug=False, num_processes=1):
    """
    """

    if not homogeneous:
        raise ValueError('Non homogeneous fitting not implemented yet!')

    # Compute the pair correlation for the true data
    # TODO: automatically calculate the range for the bins based on the data.
    bins = np.linspace(0, 10, 20)
    bin_size = bins[1] - bins[0]

    true_g_arr, bin_arr = pair_correlation(data, bins=bins, boundary=region)

    true_density = np.mean([len(d) for d in data])

    # Set up the initial guesses
    # We have 5 variables:
    # parent_mean
    # child_mean
    # child_process_scale
    # diameter
    # soft_diameter
    p0 = [None]*(5 - (1 - int(use_hard_diameter)) - (1 - int(use_soft_diameter)))

    if ("parent_mean" in initial_guess) and ("child_mean" in initial_guess):
        p0[0] = initial_guess["parent_mean"]
        p0[1] = initial_guess["parent_mean"]

    elif ("parent_mean" in initial_guess) and ("child_mean" not in initial_guess):
        p0[0] = initial_guess["parent_mean"]
        p0[1] = true_density / p0[0]

    elif ("parent_mean" not in initial_guess) and ("child_mean" in initial_guess):
        p0[1] = initial_guess["child_mean"]
        p0[0] = true_density / p0[1]

    elif ("parent_mean" not in initial_guess) and ("child_mean" not in initial_guess):
        p0[0] = p0[1] = np.sqrt(true_density)


    if "child_process_scale" not in initial_guess:
        p0[2] = 1
    else:
        p0[2] = initial_guess["child_process_scale"]

    if use_hard_diameter:

        if "diameter" in initial_guess:
            p0[3] = initial_guess["diameter"]

        else:
            p0[3] = 1

    if use_soft_diameter:

        if "soft_diameter" in initial_guess:
            p0[-1] = initial_guess["soft_diameter"]

        else:
            p0[-1] = 1

    if debug:
        print(f'Initial parameters: {p0}')

    n_sim = 10
    def optim_func(p):
        # p is our set of parameters
        if use_hard_diameter and use_soft_diameter:
            parent_mean, child_mean, child_process_scale, diameter, soft_diameter = p

        elif use_hard_diameter:
            parent_mean, child_mean, child_process_scale, diameter = p
            soft_diameter = None

        elif use_soft_diameter:
            parent_mean, child_mean, child_process_scale, soft_diameter = p
            diameter = None

        else:
            parent_mean, child_mean, child_process_scale = p
            diameter = None
            soft_diameter = None

        if debug:
            print(f'Current params: {p}')

        sim_g_arr = np.zeros((n_sim, len(bins)-1))

        for i in range(n_sim):
            cluster_data = [generate_cluster(parent_mean=parent_mean, child_mean=child_mean, diameter=diameter, soft_diameter=soft_diameter, child_process_scale=child_process_scale, region=region) for _ in data]

            g_arr, bin_arr = pair_correlation(cluster_data, bins=bins, boundary=region, processes=num_processes)
            sim_g_arr[i] = g_arr

        mean_fit_g_arr = np.mean(sim_g_arr, axis=0)

        diff_g = np.sum((mean_fit_g_arr - true_g_arr)**2)
        diff_density = (true_density - parent_mean*child_mean)**2
        
        if debug:
            print(f'Error: {diff_g}')
            print(f'Density mismatch: {diff_density}\n')

            plt.plot(bin_arr, mean_fit_g_arr)
            plt.plot(bin_arr, true_g_arr)
            plt.show()

        return diff_g

    def constraint_func(p):
        """
        Constrain the fit density to be the same as the real density.
        """
        return true_density - p[0] * p[1]

    bounds = [(0, np.inf), # parent_mean
              (0, np.inf), # child_mean
              (0, np.inf), # child_process_scale
             ]

    if use_hard_diameter:
        bounds += [(0, np.inf)]

    if use_soft_diameter:
        bounds += [(0, np.inf)]

    constraint_dict = {'type': 'eq',
                       'fun': constraint_func}

    result = minimize(optim_func, x0=p0, constraints=constraint_dict, bounds=bounds, method='COBYLA')

    return result
