#purpose: NGC5102 JAM Model
#author: Olivia Silcock
#date: Updated April 2025

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
data_path = '/Users/livisilcock/Documents/PROJECTS/NGC5102/files/JAM_NFW/'
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

# Parameters
inc0 = df["chain_median_inc"].values[0]
beta0 = df["chain_median_beta"].values[0]
bh0 = df["chain_median_mbh"].values[0] * bhm
ml0 = df["chain_median_ml"].values[0]
r_nfw = df["chain_median_r_nfw"].values[0]
rho_nfw = df["chain_median_rho_nfw"].values[0]

# Errors
inc1 = df["lower_error_inc"].values[0]
inc2 = df["upper_error_inc"].values[0]

beta1 = df["lower_error_beta"].values[0]
beta2 = df["upper_error_beta"].values[0]

bh1 = df["lower_error_mbh"].values[0] * bhm
bh2 = df["upper_error_mbh"].values[0] * bhm

ml1 = df["lower_error_ml"].values[0]
ml2 = df["upper_error_ml"].values[0]

r1 = df["lower_error_r_nfw"].values[0]
r2 = df["upper_error_r_nfw"].values[0]

rho1 = df["lower_error_rho_nfw"].values[0]
rho2 = df["upper_error_rho_nfw"].values[0]


#CALCULATIONS====================================
kin_pa, vel_offset, velsyst = fit_kinematic_pa(x=xbin, y=ybin, vel=vel, dvel=dvel, plot=False)

# flux image
flux_image = "/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/NGC5102_flux_image.fits"
f = fits.open(flux_image)
header = f[0].header
sigma_value = header['SEEING']


#RUN MODEL========================================
plt.figure(figsize=(15, 15))

jam = jam_axi_proj(
	surf_lum, sigma_lum, qObs_lum, 
	surf_pot, sigma_pot, qObs_pot,
	inc0, bh0, dist, rot_x, rot_y, 
	align='cyl', moment='zz', plot=True, pixsize=pixsize,
	quiet=1, sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins,
	beta=numpy.full_like(qObs_lum, beta0), 
	data=rms, errors=erms, ml=ml0
	# If your version of JAM supports NFW, you could add r_nfw=r_nfw, rho_nfw=rho_nfw here
)

jam.model
print(jam.chi2)


#PLOT=============================================
vmin = numpy.nanmin(rms)
vmax = numpy.nanmax(rms)

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

plt.sca(ax1)
im = display_bins(x=xpix, y=ypix, bin_num=BinNum, vel_bin=rms, cmap='twilight', pixelsize=0.2, vmin=vmin, vmax=vmax)
ax1.set_ylabel("$y\, \, [\mathrm{arcsec}]$")
ax1.set_xlabel("$x\, \, [\mathrm{arcsec}]$")

plt.sca(ax2)
im = display_bins(x=xpix, y=ypix, bin_num=BinNum, vel_bin=jam.model, cmap='twilight', pixelsize=0.2, vmin=vmin, vmax=vmax)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size='5%', pad=0.1)
cax.set_title("$V_{\mathrm{rms}}\,\,[km/s]$")
ax2.set_xlabel("$x\, \, [\mathrm{arcsec}]$")
ax2.annotate("%.2f" % jam.chi2, (-15, 15), fontsize=16)
cbar = plt.colorbar(im, cax=cax)

plt.show()








residuals = rms - jam.model
fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

# Data
plt.sca(ax1)
im1 = display_bins(x=xpix, y=ypix, bin_num=BinNum, vel_bin=rms,
                   cmap='twilight', pixelsize=0.2, vmin=vmin, vmax=vmax)
ax1.set_title("Observed $V_{rms}$")

# Model
plt.sca(ax2)
im2 = display_bins(x=xpix, y=ypix, bin_num=BinNum, vel_bin=jam.model,
                   cmap='twilight', pixelsize=0.2, vmin=vmin, vmax=vmax)
ax2.set_title("Model $V_{rms}$")

# Residuals
plt.sca(ax3)
im3 = display_bins(x=xpix, y=ypix, bin_num=BinNum, vel_bin=residuals,
                   cmap='coolwarm', pixelsize=0.2, vmin=-10, vmax=10)
ax3.set_title("Residuals (Data - Model)")

plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
