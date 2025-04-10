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

# MGE potential from NFW profile
def mge_pot(Rs, nfw_amp, arcsec_to_pc):
    r_max = max(500, 1.2 * Rs)
    r = np.linspace(0.1, r_max, 50)
    r_parsec = r * arcsec_to_pc

    R = r_parsec / Rs
    intrinsic_density = nfw_amp / (R * (1 + R) ** 2)

    assert r_parsec.shape == intrinsic_density.shape, "Shape mismatch in mge_pot inputs"

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
    sigma = p.sol[1, :] / arcsec_to_pc
    qobs = np.ones_like(surf)
    return surf, sigma, qobs

# JAM likelihood with NFW model
def jam_nfw_lnprob(pars):
    inc, beta, mbh, ml, log10_Rs, log10_nfw_amp = pars

    # Bounds check
    if not (d['inc_bounds'][0] < inc < d['inc_bounds'][1]):
        return -np.inf
    if not (d['beta_bounds'][0] < beta < d['beta_bounds'][1]):
        return -np.inf
    if not (d['mbh_bounds'][0] < mbh < d['mbh_bounds'][1]):
        return -np.inf
    if not (d['ml_bounds'][0] < ml < d['ml_bounds'][1]):
        return -np.inf
    if not (d['Rs_bounds'][0] < log10_Rs < d['Rs_bounds'][1]):
        return -np.inf
    if not (d['p0_bounds'][0] < log10_nfw_amp < d['p0_bounds'][1]):
        return -np.inf

    # Convert from log10 space
    Rs = 10 ** log10_Rs
    nfw_amp = 10 ** log10_nfw_amp

    try:
        surf_dm, sigma_dm, qobs_dm = mge_pot(Rs, nfw_amp, d['arcsec_to_pc'])
        combined_surface_density = np.concatenate((d['surf_pot'], surf_dm))
        combined_sigma = np.concatenate((d['sigma_pot'], sigma_dm))
        combined_q = np.concatenate((d['qObs_pot'], qobs_dm))
    except Exception as e:
        print(f"Error in MGE construction: {e}")
        return -np.inf

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
    return chi2 if np.isfinite(chi2) else -np.inf

# MCMC runner
def run_mcmc_nfw(output_path, ndim, nwalkers, nsteps):
    initial_pos = [
        [88, -0.1, 1.0, 3.3, 3.25, -0.75] + 0.01 * np.random.randn(ndim)
        for _ in range(nwalkers)
    ]

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print("Starting MCMC...")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, jam_nfw_lnprob, pool=pool)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    print("Saving results...")
    with open(output_path, "wb") as f:
        pickle.dump(sampler, f)

# MAIN ======================
if __name__ == "__main__":
    output_path = "/fred/oz059/olivia/NFW_samples.pkl"
    ndim = 6
    nwalkers = 20
    nsteps = 500

    with open("/home/osilcock/DM_NFW_data/kwargs.pkl", "rb") as f:
        d = pickle.load(f)

    print(f"Running EMCEE with {nsteps} steps, {nwalkers} walkers, {ndim} parameters.")
    run_mcmc_nfw(output_path, ndim, nwalkers, nsteps)
    print("MCMC sampling completed and saved!")
