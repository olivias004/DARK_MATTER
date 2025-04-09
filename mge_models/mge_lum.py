
# Name: final gravitational potential model
# Author: Olivia Silcock
# Date: October 2024

#IMPORTS=====================
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
from pylab import figure, cm
import copy


import mgefit
from mgefit.find_galaxy import find_galaxy
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
from mgefit.mge_print_contours import mge_print_contours
from jampy.mge_radial_mass import mge_radial_mass

from astropy.coordinates import Angle
from astropy.convolution import convolve
from astropy.io import fits
from astropy.modeling.models import Ellipse2D
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.visualization import (ZScaleInterval,MinMaxInterval,SqrtStretch,ImageNormalize)

from photutils.datasets import make_100gaussians_image
from photutils.segmentation import (detect_sources,
                                    make_2dgaussian_kernel)
from photutils.segmentation import SourceCatalog

from scipy import special
#CONSTANTS===================
data_path = "/Users/livisilcock/Documents/PROJECTS/NGC5102/"
EXPTIME = 460  # Exposure time in seconds
LEGACY_SCALE = 0.262
HST_SCALE = 0.1  # HST pixel scale in arcseconds per pixel
scale_factor = 0.262 / 0.1 # Legacy scale / HST scale
distance = 4e6  # in pc
pc = distance * np.pi/64800

# normPSF = [1]#countsPSF / np.sum(countsPSF)
# sigmaPSF = [0]

#FUNCTIONS===================
def _gauss2d_mge(n, xc, yc, sx, sy, pos_ang):
    ang = np.radians(pos_ang - 90.)
    x, y = np.ogrid[-xc:n[0] - xc, -yc:n[1] - yc]
    xcosang = np.cos(ang) / (np.sqrt(2.) * sx) * x
    ysinang = np.sin(ang) / (np.sqrt(2.) * sx) * y
    xsinang = np.sin(ang) / (np.sqrt(2.) * sy) * x
    ycosang = np.cos(ang) / (np.sqrt(2.) * sy) * y
    im = (xcosang + ysinang)**2 + (ycosang - xsinang)**2
    return np.exp(-im)

def _multi_gauss(pars, img, sigmaPSF, normPSF, xpeak, ypeak, theta):
    lum, sigma, q = pars
    u = 0.
    for lumj, sigj, qj in zip(lum, sigma, q):
        sx = np.sqrt(sigj**2)
        sy = np.sqrt((sigj * qj)**2)
        g = _gauss2d_mge(img.shape, xpeak, ypeak, sx, sy, theta)
        u += lumj  * g
    sx = np.sqrt(sigma**2 + sigmaPSF[:, None]**2)
    sy = np.sqrt((sigma * q)**2 + sigmaPSF[:, None]**2)
    #u[round(xpeak), round(ypeak)] = (lum * normPSF[:, None] * special.erf(2**-1.5 / sx) * special.erf(2**-1.5 / sy)).sum()
    return u

#LOAD IN DATA================
#Load PSF
pathname = data_path + "fits/PSF_MGE_5.fits"
PSF_data = fits.open(pathname)[1].data # the table data

sigmaPSF = PSF_data.sigma_pix #pixels
countsPSF = PSF_data.enc_counts
normPSF = countsPSF/np.sum(countsPSF)


#Load Legacy
legacy_path = data_path + "fits/cutout_200.4900_-36.6302.fits"
legacy_data = fits.open(legacy_path)[0].data[1, :, :]

find_result = find_galaxy(legacy_data, plot=False)
eps = find_result.eps
pa = find_result.pa
theta = find_result.theta
xpeak = find_result.xpeak
ypeak = find_result.ypeak


# Load Mitzkus data
mitzkus_data_path = data_path + "files/lum_mge_5102_mitzkus.dat"
mitzkus_data = np.loadtxt(mitzkus_data_path)

surface_density = mitzkus_data[0, :] #M_sun/pc^2
sigma_arcsec = mitzkus_data[1, :] #arcseconds
q = mitzkus_data[2, :]


#LEGACY PREPARATION==========
#Stats
mean, median, std = sigma_clipped_stats(legacy_data)
background_subtracted_image = legacy_data-median
threshold = 3.*std
kernel = make_2dgaussian_kernel(10, size=13)
convolved_data = convolve(legacy_data, kernel)
segm = detect_sources(convolved_data, threshold, npixels=10)

#plot
fig, [ax1,ax2] = plt.subplots(nrows=1,ncols=2)
ax1.imshow(convolved_data,origin='lower', vmin=0.2,vmax=2)
ax2.imshow(segm,origin='lower')
plt.show()

segm_array = segm.data #access segmentation map array (https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SegmentationImage.html#photutils.segmentation.SegmentationImage.make_source_mask)
cat = SourceCatalog(legacy_data, segm, convolved_data=convolved_data) # construct a catalogue from the segmentation map

flux = cat.kron_flux #to access flux
idx = np.where(flux == np.nanmax(flux)) #index for where flux is largest
gal_seg_idx = cat.labels[idx] #corresponding segmentation index


segm.remove_labels(labels=gal_seg_idx[0])
mask = segm.make_source_mask()
background_subtracted_image[np.isnan(background_subtracted_image)] = 0  #Dealing with background and NaNs

img = background_subtracted_image
mask = mask

#Sectors photometry
s = sectors_photometry(img, eps, pa, xpeak, ypeak,
                       minlevel=0.010, plot= True, mask=~mask)

# mge fit sectors
m = mge_fit_sectors(radius = s.radius, angle = s.angle, counts = s.counts, eps = eps,
                    ngauss=900, sigmapsf=sigmaPSF, normpsf=normPSF,
                    scale=LEGACY_SCALE, plot= True, bulge_disk=0, linear=True
                    ,outer_slope=4)

plt.figure(figsize=(8,8))  
plt.show()

#Post-processing
total_counts_LEGACY = m.sol[0,:]
surf_LEGACY = m.sol[0, :] / (2. * np.pi * m.sol[1, :]**2 * m.sol[2, :])
sigma_LEGACY = m.sol[1, :] * LEGACY_SCALE * scale_factor #arcseconds
qobs_LEGACY = m.sol[2, :]

# Print results in HST scale
print("Total Counts (HST):", total_counts_LEGACY)
print("Surface Density (HST):", surf_LEGACY)
print("Sigma (HST arcseconds):", sigma_LEGACY)
print("Axis Ratios (HST):", qobs_LEGACY)

#HST PREPARATION=============
# image
image_size = (2000, 2000)
x_center, y_center = image_size[0]//2,image_size[1]//2
model_image = np.zeros(image_size)

model_image = _multi_gauss((surface_density, sigma_arcsec/HST_SCALE, q), model_image, np.array([0,0]), np.array([1,1]), x_center, y_center, theta)

# Plot model image
plt.figure(figsize=(8, 8))
plt.imshow(np.log10(model_image + 1), origin='lower', cmap='viridis')
plt.colorbar(label='log(Intensity)')
plt.title('Mitzkus Model in HST Pixels with Rotation from LEGACY Data')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.show()

# Sectors photometry
s_HST = sectors_photometry(model_image, eps, pa, x_center, y_center, minlevel=0, plot=True)
plt.show()

# MGE fit
m = mge_fit_sectors(
    radius=s_HST.radius,
    angle=s_HST.angle,
    counts=s_HST.counts,
    eps=eps,
    ngauss=12,
    scale=HST_SCALE,
    plot=True
    )

plt.show()

# MGE radial plot comparison
rad = np.linspace(0, 500, 1000) 
dist = 4
inc = 87.5

surf_HST = m.sol[0, :] / (2. * np.pi * m.sol[1, :]**2 * m.sol[2, :]) #total counts to 
sigma_HST = m.sol[1, :] * HST_SCALE #arcseconds
qobs_HST = m.sol[2, :]


og_mass = mge_radial_mass(surface_density, sigma_arcsec, q, inc, rad, dist)
my_mass = mge_radial_mass(surf_HST, sigma_HST, qobs_HST, inc, rad, dist)

plt.scatter(rad, og_mass, label='og_mass')
plt.scatter(rad, my_mass, label='my_mass')
plt.legend()
plt.xlabel('radius (100 arcsec)')
plt.ylabel('lum')
plt.show()



# COMBINING DATA=============
# Use the radii and counts from the sectors photometry results
legacy_radii = s.radius * LEGACY_SCALE  # Scale Legacy radius to arcseconds if needed
hst_radii = s_HST.radius * HST_SCALE    # HST radius already in arcseconds


# Plot
plt.scatter(legacy_radii, np.log10(s.counts*500), label='Legacy Counts', color='blue', alpha=0.6)
plt.scatter(hst_radii, np.log10(s_HST.counts), label='HST Counts', color='red', alpha=0.6)
plt.xlabel('Radius (arcseconds)')
plt.ylabel('Counts')
#plt.ylim(0,0.3e6)
plt.legend()
plt.title('Counts vs. Radius for Legacy and HST Data')
plt.show()


# Clipping
HST_clip = 50
LEGACY_clip = 40

HST_mask = (hst_radii < HST_clip)
LEGACY_mask = (legacy_radii > LEGACY_clip)



#Snipping
H_angle_clipped  = s_HST.angle[HST_mask]
H_radius_clipped = hst_radii[HST_mask]
H_counts_clipped = s_HST.counts[HST_mask]

L_angle_clipped  = s.angle[LEGACY_mask]
L_radius_clipped = legacy_radii[LEGACY_mask]
L_counts_clipped = s.counts[LEGACY_mask]

# Plot
plt.scatter(H_radius_clipped,np.log10(H_counts_clipped),color = 'k', marker = 'x', label = 'HST')
plt.scatter(L_radius_clipped,np.log10(L_counts_clipped * 500),color = 'r', marker = 'o', label = 'LEGACY')
plt.xlabel('Radius (arcseconds)')
plt.ylabel('Counts')
plt.legend()
plt.show()

# Concatenate datasets
total_radius = np.concatenate((L_radius_clipped, H_radius_clipped))
total_counts = np.concatenate((L_counts_clipped * 500, H_counts_clipped))
total_angles = np.concatenate((L_angle_clipped, H_angle_clipped))

# Plot
plt.scatter(total_radius,np.log10(total_counts),color = 'k', marker = 'x',)
plt.xlabel('Radius (arcseconds)')
plt.ylabel('Counts')
plt.title ('Counts vs Radius: Datasets Combined')
plt.show()


#FIT CHECK=====================
m = mge_fit_sectors(radius = total_radius/0.1, angle = total_angles, counts = total_counts, eps = eps,
                    ngauss=900, sigmapsf=sigmaPSF, normpsf=normPSF,
                     plot= True, bulge_disk=0, linear=True
                    ,outer_slope=4)

plt.show()

# mge radial
rad = np.linspace(0, 500, 1000) 
dist = 4
inc = 87.5

surf_total = m.sol[0, :] / (2. * np.pi * m.sol[1, :]**2 * m.sol[2, :]) #total counts to 
sigma_total = m.sol[1, :] * HST_SCALE #arcseconds
qobs_total = m.sol[2, :]


og_mass = mge_radial_mass(surface_density, sigma_arcsec, q, inc, rad, dist)
my_mass = mge_radial_mass(surf_total, sigma_total, qobs_total, inc, rad, dist)

plt.scatter(rad, og_mass, label='og_mass')
plt.scatter(rad, my_mass, label='my_mass')
plt.legend()
plt.xlabel('radius (100 arcsec)')
plt.ylabel('lum')
plt.show()


#FITS FILE===================
from astropy.io import fits
import numpy as np

# Header information
header1 = fits.Header()
header1.append(('Dataset', 'Combined L+HST'))
header1.append(('Band', 'r'))
header1.append(('Galaxy', 'NGC5102'))
header1.append(('xpeak', xpeak, "pixels"))
header1.append(('ypeak', ypeak, "pixels"))
header1.append(('theta', theta, "degrees"))
header1.append(('PA', pa, "degrees"))
header1.append(('Eps', eps))
header1.append(('Lscale', LEGACY_SCALE, "arc/pix"))
header1.append(('HSTscale', HST_SCALE, 'arc/pix'))

# Create the primary HDU (empty primary image)
primary_hdu = fits.PrimaryHDU(header=header1)

# Sectors information table
a1 = fits.Column(name='angle', format='D', unit='degrees', array=total_angles)
a2 = fits.Column(name='radius', format='D', unit='pixels', array=total_radius)
a3 = fits.Column(name='counts', format='D', unit='counts/pixel', array=total_counts)
loc_table_1 = fits.BinTableHDU.from_columns([a1, a2, a3], name='sectors')

# PSF information table
b1 = fits.Column(name='surf', format='D', unit='counts', array=surf_total)
b2 = fits.Column(name='sigma', format='D', unit='arcsec', array=sigma_total)
b3 = fits.Column(name='qObs', format='D', array=qobs_total)
loc_table_2 = fits.BinTableHDU.from_columns([b1, b2, b3], name='mge')

# Write to a FITS file
hdulist = fits.HDUList([primary_hdu, loc_table_1, loc_table_2])
hdulist.writeto('/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/final_lum.fits', overwrite=True)





