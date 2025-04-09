# Purpose: NGC5102_JAM with Mitzkus Code Integration and 1D NFW Model
# Author: Olivia Silcock
# Date: Mar 2025

# IMPORTS ===================
import numpy as np
import emcee
import pickle
from jampy.jam_axi_proj import jam_axi_proj
from mgefit.mge_fit_1d import mge_fit_1d
from schwimmbad import MPIPool
import sys

# FUNCTIONS =================
# Log-normal prior helper
# def prior_log_normal(x, mu, sigma):
#     if x <= 0:
#         return -np.inf
#     return -0.5 * ((np.log(x) - mu) / sigma) ** 2

# mge pot
def mge_pot(Rs, p0, arcsec_to_pc):
    r_max = max(500, 1.2 * Rs)  # Dynamically adjust max radius
    r = np.linspace(0.1, r_max, 50)  # Linear sampling

    # Convert radius to parsecs
    r_parsec = r * d['arcsec_to_pc']

    # NFW density profile
    R = r_parsec / Rs
    intrinsic_density = p0 / (R * ((1 + R) ** 2))

    # Fit 1D MGE
    p = mge_fit_1d(
        r_parsec, intrinsic_density,
        negative=False,
        ngauss=11,
        rbounds=None,
        inner_slope=1,
        outer_slope=3,
        quiet=False,
        plot=True
    )

    surf = p.sol[0, :]
    sigma = p.sol[1, :] / d['arcsec_to_pc']
    qobs = np.ones_like(surf)
    return surf, sigma, qobs

# JAM likelihood with NFW
def jam_nfw_lnprob(pars):
    inc, beta, mbh, ml, Rs, p0 = pars

    # Bound checks
    if not (d['inc_bounds'][0] < inc < d['inc_bounds'][1]):
        return -np.inf
    if not (d['beta_bounds'][0] < beta < d['beta_bounds'][1]):
        return -np.inf
    if not (d['mbh_bounds'][0] < mbh < d['mbh_bounds'][1]):
        return -np.inf
    if not (d['ml_bounds'][0] < ml < d['ml_bounds'][1]):
        return -np.inf
    if not (d['Rs_bounds'][0] < Rs < d['Rs_bounds'][1]):
        return -np.inf
    if not (d['p0_bounds'][0] < p0 < d['p0_bounds'][1]):
        return -np.inf

    # # Log-normal priors for Rs and p0
    # ln_prior = (
    #     prior_log_normal(Rs, mu=np.log(2000), sigma=0.5) +
    #     prior_log_normal(p0, mu=np.log(0.2), sigma=1.0)
    # )

    # if not np.isfinite(ln_prior):
    #     return -np.inf

    # DM potential component
    try:
        surf_dm, sigma_dm, qobs_dm = mge_pot(Rs, p0, d['arcsec_to_pc'])
        combined_surface_density = np.concatenate((d['surf_pot'], surf_dm))
        combined_sigma = np.concatenate((d['sigma_pot'], sigma_dm))
        combined_q = np.concatenate((d['qObs_pot'], qobs_dm))
    except Exception as e:
        print(f"Error in MGE construction: {e}")
        return -np.inf

    # Run JAM model
    jam_result = jam_axi_proj(
        d["surf_lum"],
        d["sigma_lum"],
        d["qObs_lum"],
        combined_surface_density * ml,
        combined_sigma,
        combined_q,
        inc,
        mbh * d["bhm"],
        d["dist"],
        d["rot_x"],
        d["rot_y"],
        align="cyl",
        moment="zz",
        plot=False,
        pixsize=d["pixsize"],
        quiet=1,
        sigmapsf=d["sigmapsf"],
        normpsf=d["normpsf"],
        goodbins=d['goodbins'],
        beta=np.full_like(d["qObs_lum"], beta),
        data=d['rms'],
        errors=d['erms'],
        ml=1
    )

    chi2 = -0.5 * jam_result.chi2 * len(d['rms'])

    if not np.isfinite(chi2):
        return -np.inf

    return chi2

# MCMC runner
def run_mcmc_nfw(output_path, ndim, nwalkers, nsteps):
    p0 = [
        [88, -0.1, 1.0, 3.3, 4.0, -0.75] + 0.01 * np.random.randn(ndim)
        for _ in range(nwalkers)
    ]

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print("Starting MCMC...")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, jam_nfw_lnprob, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    print("Saving results...")
    with open(output_path, "wb") as f:
        pickle.dump(sampler, f)

# MAIN ======================
if __name__ == "__main__":
    output_path = "/fred/oz059/olivia/NFW_samples.pkl"
    ndim = 6
    nwalkers = 20        # Slightly more walkers = better exploration
    nsteps = 500


    with open("/home/osilcock/DM_NFW_data/kwargs.pkl", "rb") as f:
        d = pickle.load(f)

    print(f"Running EMCEE with {nsteps} steps, {nwalkers} walkers, {ndim} parameters.")
    run_mcmc_nfw(output_path, ndim, nwalkers, nsteps)
    print("MCMC sampling completed and saved!")



import pickle
import numpy as np

# Load sampler
with open("/Users/livisilcock/Documents/PROJECTS/DARK_MATTER/files/second_model/NFW_samples.pkl", "rb") as f:
    sampler = pickle.load(f)

# Load the data dictionary (assumes it's still valid in your context)
with open("/Users/livisilcock/Documents/PROJECTS/DARK_MATTER/files/second_model/kwargs.pkl", "rb") as f:
    d = pickle.load(f)

# Extract flattened chain (remove burn-in if needed)
flat_samples = sampler.get_chain(discard=100, thin=10, flat=True)

# Log-probability function (you can re-import or redefine if needed)
def jam_nfw_lnprob(pars):
    inc, beta, mbh, ml, Rs, p0 = pars

    if not (d['inc_bounds'][0] < inc < d['inc_bounds'][1]):
        return -np.inf
    if not (d['beta_bounds'][0] < beta < d['beta_bounds'][1]):
        return -np.inf
    if not (d['mbh_bounds'][0] < mbh < d['mbh_bounds'][1]):
        return -np.inf
    if not (d['ml_bounds'][0] < ml < d['ml_bounds'][1]):
        return -np.inf
    if not (d['Rs_bounds'][0] < Rs < d['Rs_bounds'][1]):
        return -np.inf
    if not (d['p0_bounds'][0] < p0 < d['p0_bounds'][1]):
        return -np.inf

    try:
        r_max = max(500, 1.2 * Rs)
        r = np.linspace(0.1, r_max, 50)
        r_parsec = r * d['arcsec_to_pc']
        R = np.clip(r_parsec / Rs, 1e-6, None)
        intrinsic_density = p0 / (R * (1 + R)**2)

        from mgefit.mge_fit_1d import mge_fit_1d
        p = mge_fit_1d(r_parsec, intrinsic_density, negative=False, ngauss=11,
                       inner_slope=1, outer_slope=3, quiet=True)

        surf_dm, sigma_dm = p.sol[0], p.sol[1] / d['arcsec_to_pc']
        qobs_dm = np.ones_like(surf_dm)

        combined_surf = np.concatenate((d['surf_pot'], surf_dm))
        combined_sigma = np.concatenate((d['sigma_pot'], sigma_dm))
        combined_q = np.concatenate((d['qObs_pot'], qobs_dm))

        from jampy.jam_axi_proj import jam_axi_proj
        jam_result = jam_axi_proj(
            d["surf_lum"],
            d["sigma_lum"],
            d["qObs_lum"],
            combined_surf * ml,
            combined_sigma,
            combined_q,
            inc,
            mbh * d["bhm"],
            d["dist"],
            d["rot_x"],
            d["rot_y"],
            align="cyl",
            moment="zz",
            plot=False,
            pixsize=d["pixsize"],
            quiet=1,
            sigmapsf=d["sigmapsf"],
            normpsf=d["normpsf"],
            goodbins=d['goodbins'],
            beta=np.full_like(d["qObs_lum"], beta),
            data=d['rms'],
            errors=d['erms'],
            ml=1
        )

        chi2 = -0.5 * jam_result.chi2 * len(d['rms'])
        return chi2 if np.isfinite(chi2) else -np.inf

    except Exception as e:
        print(f"Error in chi2 eval: {e}")
        return -np.inf

# Compute chi-squared values
lnprobs = np.array([jam_nfw_lnprob(sample) for sample in flat_samples])

# Convert to actual chi2 from log-prob if desired
chi2_vals = -2 * lnprobs  # because lnprob = -0.5 * chi2 * N

# Inspect
print("Chi-squared values (first 20):")
print(chi2_vals[:20])

print(f"\nSummary:\nMin: {np.nanmin(chi2_vals)}\nMax: {np.nanmax(chi2_vals)}\nMean: {np.nanmean(chi2_vals)}")









