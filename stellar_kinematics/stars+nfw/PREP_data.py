#purpose: prep data
#author: Olivia Silcock
#date: Dec 2023

import pickle
import numpy
from astropy.io import fits
from math import acos
from pafit.fit_kinematic_pa import fit_kinematic_pa

#FUNCTIONS=======================================
def rotate_vectors(x,y,pa):
	""" INPUTS: X: a 1D vector of xbin positions,
	barycentre, relative to galaxy centre. [arcseconds]
	Y: 1D vector of of ybin barycentre postions. [arcseconds]
	pa: Galaxy postion angle [degrees], measured from Y axis
	counter-clockwise to major axis. 
	Function: Returns the rotated arrays so that X corresponds
	to the projected major axis, and Y to the projected minor
	axis of the galaxy."""

	# rotate clockwise by position angle
	theta = -numpy.radians(pa)
	x2 = x*numpy.cos(theta) - y*numpy.sin(theta)
	y2 = x*numpy.sin(theta) + y*numpy.cos(theta)
	
	# rotate counterclockwise by 90 degrees
	xrot = x2*numpy.cos(numpy.pi/2) - y2*numpy.sin(numpy.pi/2)
	yrot = x2*numpy.sin(numpy.pi/2) + y2*numpy.cos(numpy.pi/2)

	return xrot, yrot

def prep_data(lum_mge_path, pot_mge_path, kin_data_path, output_path):
	#constants
	dist = 4  # distance in Mpc from Earth value researched
	sigmapsf = [0.697544735125434/2.355] # sigma PSF in arcsec. 0.69 number is from flux image header. 2.355 is the conversion from FWHM to sigma
	normpsf = [1.0] # PSF weight
	bhm = (1.9*(50/200)**5.1)*1e8 #solar masses: msigma relation - https://ui.adsabs.harvard.edu/abs/2011Natur.480..215M/abstract
	Rs_range = (100,2000,15)
	p0_range = (0.001, 2.0, 15)
	arcsec_to_pc = 4e6 / 206265

	#lum_mge_path (MGE from MUSE field)
	lum_hdu = fits.open(lum_mge_path)

	surf_lum = lum_hdu['mge'].data.surf
	sigma_lum = lum_hdu['mge'].data.sigma
	qObs_lum = lum_hdu['mge'].data.qObs

	incmin = (acos(numpy.min(qObs_lum)))*(180/numpy.pi) +5 # degrees

	lum_hdu.close()

	#pot_mge_pth (MGE based on Mitzkus mass model)
	pot_hdu = fits.open(pot_mge_path)

	surf_pot = pot_hdu['mge'].data.surf
	sigma_pot = pot_hdu['mge'].data.sigma #arcseconds
	qObs_pot = pot_hdu['mge'].data.qObs

	pot_hdu.close()

	#from kin_data_path
	kinematics_hdu = fits.open(kin_data_path)

	pixsize = kinematics_hdu[0].header['PIXSCALE']

	xbin = kinematics_hdu['RESULTS'].data.xbar #arcseconds
	ybin = kinematics_hdu['RESULTS'].data.ybar #arcseconds
	vel = kinematics_hdu['RESULTS'].data.velbin #km/s
	dvel = kinematics_hdu['RESULTS'].data.dvelbin #km/s
	sig = kinematics_hdu['RESULTS'].data.sigbin #km/s
	dsig = kinematics_hdu['RESULTS'].data.dsigbin #km/s

	xpix = kinematics_hdu['COORDINATES'].data.xpix #arcseconds
	ypix = kinematics_hdu['COORDINATES'].data.ypix #arcseconds
	BinNum = kinematics_hdu['COORDINATES'].data.binNum.astype(int)

	goodbins = numpy.isfinite(xbin)  # Here we fit all bins
	vel = vel-numpy.median(vel)
	rms = numpy.sqrt(vel**2 + sig**2) #observed
	erms = numpy.sqrt((dvel*vel)**2+(dsig*sig)**2)/rms #error


	kin_pa, vel_offset, velsyst = fit_kinematic_pa(x = xbin, y = ybin, vel = vel, dvel = dvel, plot = False)
	rot_x, rot_y = rotate_vectors(xbin, ybin, kin_pa)

	kinematics_hdu.close()

	#bounds
	inc_bounds = [incmin, 90]
	beta_bounds = [-0.99, 0.99]
	mbh_bounds = [0.8, 1.2]
	ml_bounds = [0.1, 10]
	Rs_bounds = [3.0, 3.5]
	p0_bounds = [-2.5,0]


	#pulling in all the below values
	kwargs = {'surf_lum': surf_lum, #M_sun/pc^2
			'sigma_lum': sigma_lum, #arcseconds
			'qObs_lum': qObs_lum,
			'surf_pot': surf_pot, #M_sun/pc^2
			'sigma_pot': sigma_pot, #arcseconds
			'qObs_pot': qObs_pot, 
			'dist': dist, #Mpc
			'xbin': xbin, #arcseconds
			'ybin': ybin, #arcseconds
			'rot_x': rot_x, 
			'rot_y': rot_y,
			'sigmapsf': sigmapsf, #arcseconds
			'normpsf': normpsf,
			'rms': rms,
			'erms': erms,
			'vel': vel,
			'dvel': dvel,
			'pixsize': pixsize,
			'goodbins': goodbins,
			'plot': 0,
			'xpix': xpix,
			'ypix': ypix,
			'BinNum': BinNum,
			'inc_bounds': inc_bounds,
			'beta_bounds': beta_bounds,
			'mbh_bounds': mbh_bounds,
			'ml_bounds': ml_bounds,
			'bhm': bhm,
			'Rs_range': Rs_range,
			'p0_range': p0_range,
			'arcsec_to_pc': arcsec_to_pc,
			'Rs_bounds': Rs_bounds,
			'p0_bounds': p0_bounds
			}
	
	with open(output_path, 'wb') as f:
		pickle.dump(kwargs, f)


lum_mge_path = "/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/final_lum.fits"
pot_mge_path = "/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/final_pot.fits"
kin_data_path = "/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/NGC5102_ppxf_fitted_data_SN_200.fits"
output_path = "/Users/livisilcock/Documents/PROJECTS/NGC5102/files/JAM_NFW/kwargs.pkl"

prep_data(lum_mge_path, pot_mge_path, kin_data_path, output_path)    



