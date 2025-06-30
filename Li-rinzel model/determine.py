#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:49:44 2025

@author: deepikaneelapala
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import peakutils # type: ignore
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
import time
from scipy.stats import norm



# Load time and experimental data
#time1 = pd.read_excel('/Users/deepikaneelapala/Local_NSD/Deepika/Research/Aug_Dec2024/Connectivity Model/ISBI_2025/time.xlsx') / 1000  # Convert to seconds
#exp_data = pd.read_excel('/Users/deepikaneelapala/Local_NSD/Deepika/Research/Jan-May 2025/Intensity Measurements/Data/54_intensity.xlsx').iloc[:, 12].to_numpy()[:249]
# Normalize function
def normalisation(k_Ca):
    min_Ca = np.min(k_Ca)
    max_Ca = np.max(k_Ca)
    return (k_Ca - min_Ca) / (max_Ca - min_Ca)

# Baseline correction and normalization
'''baseline = peakutils.baseline(exp_data, deg=5)
basecor_ca = exp_data - baseline
normalised_exp = normalisation(basecor_ca).reshape(-1, 1)'''
time1 = np.arange(0, 300)
def lirinzel_ns(v2, a2, IP3, d5, k3):
    c0, c1 = 2, 0.185
    v1, v3 = 6, 0.9
    d1, d2, d3, Nd = 0.13, 1.049, 0.9434, 2
    time = np.arange(0, 250, 0.1)
    dt = 0.1
    Ca, h = [0.1], [0.1]  # Initialize lists

    # ODE solving
    for _ in range(1, len(time)):  # Start from index 1 to avoid index errors
        CaER = (c0 - Ca[-1]) / c1
        pinf = IP3 / (IP3 + d1)
        ninf = Ca[-1] / (Ca[-1] + d5)
        Q2 = d2 * (IP3 + d1) / (IP3 + d3)
        tauh = 1 / (a2 * (Q2 + Ca[-1]))
        hinf = Q2 / (Q2 + Ca[-1])
        alphah = hinf / tauh
        betah = (1 - hinf) / tauh
        dh = alphah * (1 - h[-1]) - betah * h[-1]
        dCa = (c1 * v1 * pinf**3 * ninf**3 * h[-1]**3 * (CaER - Ca[-1]) +
               c1 * v2 * (CaER - Ca[-1]) -
               v3 * Ca[-1]**2 / (Ca[-1]**2 + k3**2))
        
        # Append new values instead of assigning to an index
        Ca.append(Ca[-1] + dCa * dt)
        h.append(h[-1] + dh * dt)


    # Convert to NumPy array for interpolation
    Ca = np.array(Ca)

    # Interpolation to match experimental time points
    inter_func = interp1d(time1, Ca, fill_value="extrapolate")
    Ca_sim = inter_func(time1).flatten()
    Ca_sim = normalisation(Ca_sim)

    # Handle NaN values
    Ca_sim = np.nan_to_num(Ca_sim)

    return Ca_sim


# Optimization problem definition
class CalciumModelProblem(Problem):
    def __init__(self):
        super().__init__(n_var=5, n_obj=1, xl=[0, 0, 0.4, 0.08,0], xu=[0.1, 0.25, 0.6, 0.12,0.1])
        #super().__init__(n_var=4, n_obj=1, xl=[5, 0.4, 0.435, 0.75], xu=[16, 0.8, 1, 1.7])

    '''def _evaluate(self, x, out, *args, **kwargs):
        mse_per_simulation = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            v2, a2, IP3, d5, k3 = x[i]
            Ca_simulated = lirinzel_ns(v2, a2, IP3, d5, k3)
            mse_per_simulation[i] = mean_squared_error(Ca_simulated, normalised_exp.flatten())
        out["F"] = mse_per_simulation'''

# Run optimization
start_time = time.time()
problem = CalciumModelProblem()
algorithm = GA(pop_size=100, eliminate_duplicates=True)
res = minimize(problem, algorithm, seed=1, verbose=True)
end_time = time.time()

# Results
print(f"Execution time: {end_time - start_time:.2f} seconds")
print(f"Best solution: {res.X}")
#print(f"Best MSE: {res.F[0]}")

# Plot results
best_v2, best_a2, best_IP3, best_d5,best_k3 = res.X
Ca_simulated_best = lirinzel_ns(best_v2, best_a2, best_IP3, best_d5,best_k3)
plt.figure(figsize=(10, 6))
#plt.plot(time1, normalised_exp, label="Experimental Data", color="black", linestyle ='--',linewidth =2)
plt.plot(time1, Ca_simulated_best, label=f"Simulated Data\n(v2={best_v2:.2f}, a2={best_a2:.2f}, IP3={best_IP3:.2f}, d5={best_d5:.2f},k3={best_k3:.2f})", color="red",linewidth =2)
plt.legend()
plt.title("Experimental vs Simulated Calcium Concentrations")
plt.xlabel("Time (s)")
plt.xlim(0,250)
plt.ylabel("Normalized Calcium Concentration")
plt.grid()
plt.show()
