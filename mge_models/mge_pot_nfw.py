# Name: final gravitational potential model including NFW model
# Author: Olivia Silcock
# Date: January 2025


#IMPORTS=====================
import numpy as np
import matplotlib.pyplot as plt
from mgefit.mge_fit_1d import mge_fit_1d




#DEFINE NFW MGE==============
#NFW profile
NFW_density      = 0.9
NFW_scale_radius = 205

NFW_radius = 500*10**(np.linspace(-5,1,num=1000))

DM_profile = NFW_density/((NFW_radius/NFW_scale_radius)*(1+NFW_radius/NFW_scale_radius)*(1+NFW_radius/NFW_scale_radius))


plt.scatter(NFW_radius,DM_profile, marker = '+')
plt.xlabel("log10 radius (arcseconds)")
plt.ylabel("log10 density")
plt.xscale("log")
plt.yscale("log")
plt.show()



#Fitting the MGE with the DM_Profile
from mgefit.mge_fit_1d import mge_fit_1d

# Fit the NFW profile using mge_fit_1d
p = mge_fit_1d(
    NFW_radius, 
    DM_profile, 
    negative=False, 
    ngauss=12, 
    rbounds=None, 
    inner_slope=2, 
    outer_slope=3, 
    quiet=False, 
    fignum=1, 
    plot=True
)

plt.show()

# Extract MGE parameters for the NFW profile
lum_DM = p.sol[0]  # Luminosities of each Gaussian
sigma_DM = p.sol[1]  # Sigma (radial width) of each Gaussian in arcseconds
q_DM = np.ones_like(lum_DM)  # Assuming spherical symmetry for NFW profile





og_path = "/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/final_pot.fits"
og_data = fits.open(og_path)[0].data[1, :, :]


#CONCATENATION===============
# Combine the NFW MGE with the existing gravitational potential MGE
combined_lum = np.concatenate((total_counts, lum_DM))
combined_sigma = np.concatenate((sigma_total, sigma_DM))
combined_q = np.concatenate((qobs_total, q_DM))

# Check the combined MGE profile
plt.figure(figsize=(8, 6))
plt.scatter(sigma_total, surf_total, label='Existing MGE', alpha=0.7)
plt.scatter(sigma_DM, lum_DM / (2 * np.pi * sigma_DM**2), label='NFW MGE', alpha=0.7)
plt.xlabel('Sigma (arcseconds)')
plt.ylabel('Surface Density')
plt.legend()
plt.title('Combined MGE: Existing + NFW')
plt.show()
