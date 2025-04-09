#purpose: NGC5102 JAM Model
#author: Olivia Silcock
#date: September 2023

#IMPORTS==========================================
import astropy.io.fits as fits
import numpy as numpy
import glob
from astropy.io import fits
import os
from os import path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plotbin.display_pixels import display_pixels
from plotbin.display_bins import display_bins
from plotbin.display_bins_generators import display_bins_generators
from plotbin.plot_velfield import plot_velfield
from plotbin.symmetrize_velfield import symmetrize_velfield
from mpl_toolkits.axes_grid1 import make_axes_locatable

from math import acos

import jampy as jam_package
from jampy.jam_axi_proj import jam_axi_proj

import emcee
import corner
import pandas

from plotbin.display_bins import display_bins

import pickle
import pafit
from pafit.fit_kinematic_pa import fit_kinematic_pa


#CONSTANTS======================================
#strings
data_path = '/Users/livisilcock/Documents/PROJECTS/NGC5102/files/JAM_STARS/'
kwargs = data_path + 'kwargs.pkl'
chain_results = data_path + 'chain_results.csv'

#DEFINING VALUES========================================
with open(kwargs, 'rb') as f:
    d = pickle.load(f)

#Values
surf_lum = d['surf_lum']
sigma_lum = d['sigma_lum']
qObs_lum = d['qObs_lum']

surf_pot = d['surf_pot']
sigma_pot = d['sigma_pot']
qObs_pot = d['qObs_pot']

rms = d['rms']
erms = d['erms']
vel = d['vel']
dvel = d['dvel']

sigmapsf = d['sigmapsf']
normpsf = d['normpsf']
dist = d['dist']
pixsize = d['pixsize']
goodbins = d['goodbins']

xbin = d['xbin']
ybin = d['ybin']
rot_x = d['rot_x']
rot_y = d['rot_y']
xpix = d['xpix']
ypix = d['ypix']
BinNum = d['BinNum']

bhm = d['bhm']


#BEST FIT SOLUTIONS==============================
#csv
df = pandas.read_csv(chain_results)
df.keys() # prints out what available column names there are to access data by

#Best-fit solutions
inc0 = df["chain_median_inc"].values    # Flattening (relates to inclination). Must be < min(qObs)
beta0 = df["chain_median_beta"].values  # Anisotropy parameter, typically between -0.5 and 0.5
bh0 = ((df["chain_median_mbh"].values))*bhm   # Central supermassive black hole mass.
ml0 = df["chain_median_ml"].values    # Mass-to-light ratio, Typically between 1 and 10

#Errors
inc1 = df["lower_error_inc"].values #lower error     
inc2 = df["upper_error_inc"].values #upper error    

beta1 = df["lower_error_beta"].values #lower error    
beta2 = df["upper_error_beta"].values #upper error 

bh1 = df["lower_error_mbh"].values*bhm #lower error    
bh2 = df["upper_error_mbh"].values*bhm #upper error 

ml1 = df["lower_error_ml"].values #lower error    
ml2 = df["upper_error"].values #upper error 


#CALCULATIONS====================================
#kinematics pa
kin_pa, vel_offset, velsyst = fit_kinematic_pa(x = xbin, y = ybin, vel = vel, dvel = dvel, plot = False)#flux image file
flux_image = "/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/NGC5102_flux_image.fits"
f = fits.open(flux_image)
header = f[0].header

sigma_value = header['SEEING']


#RUN MODEL========================================
# Prepare the plot size you want
plt.figure(figsize=(15,15))

jam = jam_axi_proj(surf_lum, sigma_lum, qObs_lum, surf_pot, sigma_pot, qObs_pot,
			inc0, bh0, dist, rot_x, rot_y, align='cyl', moment='zz', plot=True, pixsize=pixsize,
			quiet=1, sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins,
			beta=numpy.full_like(qObs_lum, beta0), data=rms, errors=erms, ml=1)


jam.model
print(jam.chi2)


#PLOT=============================================
vmin = numpy.nanmin(rms)
vmax = numpy.nanmax(rms)

fig, [ax1, ax2] = plt.subplots(nrows = 1, ncols = 2)

plt.sca(ax1)
im = display_bins(x = xpix, y = ypix, bin_num = BinNum, vel_bin = rms, cmap = 'twilight', pixelsize = 0.2, vmin = vmin, vmax = vmax)
ax1.set_ylabel("$y\, \, [\mathrm{arcsec}]$")
ax1.set_xlabel("$x\, \, [\mathrm{arcsec}]$")


plt.sca(ax2)
im = display_bins(x = xpix, y = ypix, bin_num = BinNum, vel_bin = jam.model, cmap = 'twilight', pixelsize = 0.2, vmin = vmin, vmax = vmax)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size = '5%', pad = 0.1)
cax.set_title("$V_{\mathrm{rms}}\,\,[km/s]$")
ax2.set_xlabel("$x\, \, [\mathrm{arcsec}]$")
ax2.annotate("%.2f"%jam.chi2, (-15, 15), fontsize = 16)
cbar = plt.colorbar(im, cax = cax)

#plt.show()
plt.savefig(data_path + 'model_comparison.png', bbox_inches = "tight")
plt.close()






