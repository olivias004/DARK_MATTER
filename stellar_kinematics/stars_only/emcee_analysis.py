#purpose: emcee analysis
#author: Olivia Silcock
#date: September 2023

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
tau0 = [35.67234921, 33.97927069, 39.61699091, 42.27054764]
labels = ["inclination", "anisotropy", "black hole mass", "mass to light ratio"]

output_path = '/Users/livisilcock/Documents/PROJECTS/NGC5102/files/JAM_STARS/'
NGC5102_samples = output_path + 'samples.pkl'


#FUNCTIONS=======================================
#determine chi2
def jam_nfw_lnprob(pars):
    """
    Args:
    pars: the starting point for the emcee process

    Outputs:
    chi2: chi-squared value for the given inputs

    """

    inc, beta, mbh, ml = pars

    check_inc = d['inc_bounds'][0] < inc < d['inc_bounds'][1]
    check_beta = d['beta_bounds'][0] < beta < d['beta_bounds'][1]
    check_mbh = d['mbh_bounds'][0] < mbh < d['mbh_bounds'][1]
    check_ml = d['ml_bounds'][0] < ml < d['ml_bounds'][1]

    if check_inc and check_beta and check_mbh and check_ml:
            # Note: surf_pot is multiplied by ml, while the keyword ml=1
            #rmsModel, ml_best, chi2dof, fluxmodel
        jam = jam_axi_proj(d['surf_lum'], 
            d['sigma_lum'], d['qObs_lum'], 
            d['surf_pot']*ml, d['sigma_pot'], 
            d['qObs_pot'], inc, mbh, d['dist'], 
            d['rot_x'], d['rot_y'], align = 'cyl', 
            moment = 'zz', plot = False, 
            pixsize = d['pixsize'], quiet = 1, 
            sigmapsf = d['sigmapsf'], 
            normpsf = d['normpsf'], 
            goodbins = d['goodbins'],
            beta = numpy.full_like(d['qObs_lum'], beta),
            data = d['rms'], errors = d['erms'], ml = 1)
        chi2 = -0.5*jam.chi2*len(d['rms'])

    else:
        chi2 = -numpy.inf

    return chi2

#READ IN PICKLE FILE=============================
#pickle
with open(NGC5102_samples, 'rb') as f:
	reader = pickle.load(f)

#analysing samples
#tau = reader.get_autocorr_time()
try:
	tau = reader.get_autocorr_time()
except:
	tau = [35.67234921, 33.97927069, 39.61699091, 42.27054764]

#burnin = int(2 * numpy.max(tau))
burnin = int(4 * numpy.max(tau))  # More conservative
thin = int(0.5 * numpy.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

# log_prob_samples = samples.get_log_prob(discard=burnin, flat=True, thin = thin)
# log_prior_samples = samples.get_blobs(discard=burnin, flat=True, thin = thin)
# Calculate tau with a fallback
try:
    tau = reader.get_autocorr_time()
    if numpy.isnan(tau).any():
        # Set to default values if tau contains NaN
        tau = [35.67234921, 33.97927069, 39.61699091, 42.27054764]
except Exception as e:
    print(f"Warning: {e}. Using default tau values.")
    tau = tau0

burnin = int(2 * numpy.max(tau))
thin = int(0.5 * numpy.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)


#ANALYSIS=================================
#percentiles
estimated_inc = numpy.percentile(samples[:, 0], [16, 50, 84])
inc_errors = numpy.diff(estimated_inc)

estimated_beta = numpy.percentile(samples[:, 1], [16, 50, 84])
beta_errors = numpy.diff(estimated_beta)

estimated_mbh = numpy.percentile(samples[:, 2], [16, 50, 84])
mbh_errors = numpy.diff(estimated_mbh)

estimated_ml = numpy.percentile(samples[:, 3], [16, 50, 84])
ml_errors = numpy.diff(estimated_ml)

#Walkers plot
n_steps, n_walkers, n_params = reader.get_chain().shape



fig, axes = plt.subplots(n_params, figsize=(10, 7), sharex=True)
labels = ["inc", "beta", "Mbh", "M/L"]

for i in range(n_params):  # Loop over each parameter
    ax = axes[i]
    for j in range(n_walkers):  # Loop over each walker
        ax.plot(reader.get_chain()[:, j, i], "k", alpha=0.3)  # Plot each walker's samples
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step number")
plt.show()
# plt.savefig(output_path + 'walkers.png', bbox_inches = "tight")


#plot corner plot: the input sample must be 1- or 2-D.
fig = corner.corner(samples, labels=labels)
# plt.close()
plt.show()
# plt.savefig(output_path + 'MCMC_corner.png', bbox_inches = "tight")
# plt.close()


log_prob_samples = reader.get_log_prob(discard=burnin,flat=True)
log_prior_samples = reader.get_blobs(discard=burnin,flat=True)
print(numpy.nanmax(log_prob_samples))
plt.plot(log_prob_samples)
plt.show()
# plt.savefig(output_path + 'log_prob.png', bbox_inches = "tight")
# plt.close()

# #csv file
data_dictionary = {'chain_median_ml': [estimated_ml[1]], 'lower_error_ml':[ml_errors[0]], 'upper_error': [ml_errors[1]],
'chain_median_inc': [estimated_inc[1]], 'lower_error_inc': [inc_errors[0]], 'upper_error_inc': [inc_errors[1]],
'chain_median_beta': [estimated_beta[1]], 'lower_error_beta': [beta_errors[0]], 'upper_error_beta': [beta_errors[1]], 
'chain_median_mbh': [estimated_mbh[1]], 'lower_error_mbh': [mbh_errors[0]], 'upper_error_mbh': [mbh_errors[1]]}

df = pandas.DataFrame(data_dictionary)
df.to_csv(output_path + 'chain_results.csv')








# Walkers plot
n_steps, n_walkers, n_params = reader.get_chain().shape

fig, axes = plt.subplots(n_params, figsize=(10, 7), sharex=True)
labels = ["inc", "beta", "Mbh", "M/L"]

for i in range(n_params):  # Loop over each parameter
    ax = axes[i]
    for j in range(n_walkers):  # Loop over each walker
        ax.plot(reader.get_chain()[:, j, i], "k", alpha=0.3)  # Plot each walker's samples
    ax.axvline(burnin, color='red', linestyle='--', label='Burn-in point')  # Add a vertical line for burn-in with label
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

# Adding a legend to the last plot which generally has more space
axes[-1].legend(loc='upper right')
axes[-1].set_xlabel("Step number")

plt.show()
# Optionally save the figure
# plt.savefig(output_path + 'walkers.png', bbox_inches = "tight")
# plt.close()








