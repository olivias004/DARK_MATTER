# Purpose: NGC5102_JAM with Mitzkus Code Integration
# Author: Olivia Silcock
# Date: Dec 2023

# IMPORTS
import numpy as np
import emcee
import pickle
from jampy.jam_axi_proj import jam_axi_proj
from schwimmbad import MPIPool
import sys

# FUNCTIONS ========================================

# Log-probability function for MCMC
def jam_lnprob(pars):
    """
    Compute the log-probability for the MCMC process.
    Args:
        pars (list): [inclination, anisotropy (beta), black hole mass, mass-to-light ratio]
    Returns:
        float: Log-probability (chi2)
    """
    inc, beta, mbh, ml = pars

    # Parameter bounds
    if not (d['inc_bounds'][0] < inc < d['inc_bounds'][1]):
        return -np.inf
    if not (d['beta_bounds'][0] < beta < d['beta_bounds'][1]):
        return -np.inf
    if not (d['mbh_bounds'][0] < mbh < d['mbh_bounds'][1]):
        return -np.inf
    if not (d['ml_bounds'][0] < ml < d['ml_bounds'][1]):
        return -np.inf

    # JAM model computation
    jam = jam_axi_proj(
        surf_lum=d['surf_lum'],
        sigma_lum=d['sigma_lum'],
        qObs_lum=d['qObs_lum'],
        surf_pot=d['surf_pot'] * ml,
        sigma_pot=d['sigma_pot'],
        qObs_pot=d['qObs_pot'],
        inc=inc,
        mbh=mbh * d['bhm'],
        dist=d['dist'],
        xbin=d['rot_x'],
        ybin=d['rot_y'],
        align="cyl",
        moment="zz",
        plot=False,
        pixsize=d['pixsize'],
        quiet=1,
        sigmapsf=d['sigmapsf'],
        normpsf=d['normpsf'],
        goodbins=d['goodbins'],
        beta=np.full_like(d['qObs_lum'], beta),
        data=d['rms'],
        errors=d['erms'],
    )
    
    # Chi-squared computation
    chi2 = -0.5 * jam.chi2 * len(d['rms'])
    return chi2


# MCMC Sampling Function
def run_mcmc(output_path, ndim=4, nwalkers=20, nsteps=5000):
    """
    Run MCMC sampling using emcee.
    Args:
        output_path (str): Filepath to save MCMC samples.
        ndim (int): Number of dimensions.
        nwalkers (int): Number of walkers.
        nsteps (int): Number of steps.
    """
    # Initialize walkers around a starting point
    p0 = [
        [87.5, 0.0, 1.0, 2.55] + 0.5 * np.random.randn(ndim) for _ in range(nwalkers)
    ]

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, jam_lnprob, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    with open(output_path, "wb") as f:
        pickle.dump(sampler, f)


# MAIN EXECUTION ===================================
if __name__ == "__main__":
    # CONSTANTS
    location = "local"  # Change to "server" if running on the server
    ndim = 4
    nwalkers = 5
    nsteps = 100

    # FILE PATHS
    if location == "server":
        data_path = "/home/osilcock/DM_data/kwargs.pkl"
        output_path = "/fred/oz059/olivia/NGC5102_samples.pkl"
    else:
        data_path = "/Users/livisilcock/Documents/PROJECTS/NGC5102/files/final_JAM/kwargs.pkl"
        output_path = "/Users/livisilcock/Documents/PROJECTS/NGC5102/files/final_JAM/NGC5102_samples.pkl"

    # LOAD INPUT DATA
    with open(data_path, "rb") as f:
        d = pickle.load(f)

    # RUN MCMC
    run_mcmc(output_path, ndim, nwalkers, nsteps)
    print("MCMC completed and saved!")
