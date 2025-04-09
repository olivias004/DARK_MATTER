#purpose: emcee analysis
#author: Olivia Silcock
#date: Updated April 2025

#IMPORTS===================================
import pickle
import emcee
import corner
import pandas
import numpy
import math as acos

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#CONSTANTS=======================================
tau0 = [35.67234921, 33.97927069, 39.61699091, 42.27054764, 40.12345678, 38.98765432]  # add default tau for new params
labels = ["inclination", "anisotropy", "black hole mass", "mass to light ratio", "nfw radius", "nfw density"]

output_path = '/Users/livisilcock/Documents/PROJECTS/NGC5102/files/JAM_NFW/'
NGC5102_samples = output_path + 'NFW_samples.pkl'


#READ IN PICKLE FILE=============================
with open(NGC5102_samples, 'rb') as f:
    reader = pickle.load(f)

# Auto-correlation time
try:
    tau = reader.get_autocorr_time()
    if numpy.isnan(tau).any():
        tau = tau0
except Exception as e:
    print(f"Warning: {e}. Using default tau values.")
    tau = tau0

burnin = int(2 * numpy.max(tau))
thin = int(0.5 * numpy.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

# ANALYSIS=======================================
percentiles = [numpy.percentile(samples[:, i], [16, 50, 84]) for i in range(6)]
errors = [numpy.diff(p) for p in percentiles]

# Walkers plot
n_steps, n_walkers, n_params = reader.get_chain().shape
fig, axes = plt.subplots(n_params, figsize=(10, 10), sharex=True)

for i in range(n_params):
    ax = axes[i]
    for j in range(n_walkers):
        ax.plot(reader.get_chain()[:, j, i], "k", alpha=0.3)
    ax.axvline(burnin, color='red', linestyle='--', label='Burn-in point')
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].legend(loc='upper right')
axes[-1].set_xlabel("Step number")
plt.show()

# Corner plot
fig = corner.corner(samples, labels=labels)
plt.show()

# Log prob plot
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True)

print("Max log probability:", numpy.nanmax(log_prob_samples))
plt.plot(log_prob_samples)
plt.xlabel("Sample Index")
plt.ylabel("Log Probability")
plt.show()

# Export CSV
data_dictionary = {
    'chain_median_inc': [percentiles[0][1]], 'lower_error_inc': [errors[0][0]], 'upper_error_inc': [errors[0][1]],
    'chain_median_beta': [percentiles[1][1]], 'lower_error_beta': [errors[1][0]], 'upper_error_beta': [errors[1][1]],
    'chain_median_mbh': [percentiles[2][1]], 'lower_error_mbh': [errors[2][0]], 'upper_error_mbh': [errors[2][1]],
    'chain_median_ml': [percentiles[3][1]], 'lower_error_ml': [errors[3][0]], 'upper_error_ml': [errors[3][1]],
    'chain_median_r_nfw': [percentiles[4][1]], 'lower_error_r_nfw': [errors[4][0]], 'upper_error_r_nfw': [errors[4][1]],
    'chain_median_rho_nfw': [percentiles[5][1]], 'lower_error_rho_nfw': [errors[5][0]], 'upper_error_rho_nfw': [errors[5][1]]
}

df = pandas.DataFrame(data_dictionary)
df.to_csv(output_path + 'chain_results.csv', index=False)
