#purpose: mge model of NGC5102 using LEGACY DR10 data
#author: Olivia Silcock
#date: April 2023

"""
INFORMATION ON THE DATASET:
This image was taken as part of the Legacy survey, specifically, DECaLS
(in Data Release 10, DR10). The DECam is an efficient option for obtaining 
photometry in the g, r and z bands. 
"""

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


#LOAD DATA===================
#General data path
data_path = "/Users/livisilcock/Documents/PROJECTS/DARK_MATTER/files/"

#Legacy data
pathname = data_path + "mge_legacy/cutout_200.4900_-36.6302.fits"
data =fits.open(pathname)[0].data[1,:,:] #r-band image

plt.imshow(data,origin='lower', vmin=0.02,vmax=2)
plt.savefig(data_path + "mge_legacy/NGC5102_image.png", bbox_inches = "tight")
plt.close()

#PSF data
pathname = data_path + "PSF/PSF_MGE_5.fits"
PSF_data = fits.open(pathname)[1].data # the table data

sigmaPSF = PSF_data.sigma_pix #pixels
countsPSF = PSF_data.enc_counts


#STATS AND CONVOLVE==========
mean, median, std = sigma_clipped_stats(data)
background_subtracted_image = data-median
threshold = 3.*std
kernel = make_2dgaussian_kernel(10, size=13)
convolved_data = convolve(data, kernel)
segm = detect_sources(convolved_data, threshold, npixels=10)

#plot
fig, [ax1,ax2] = plt.subplots(nrows=1,ncols=2)
ax1.imshow(convolved_data,origin='lower', vmin=0.2,vmax=2)
ax2.imshow(segm,origin='lower')
plt.savefig(data_path + "mge_legacy/segment_map.png", bbox_inches = "tight")
plt.close()

#segmentation array
segm_array = segm.data #access segmentation map array (https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SegmentationImage.html#photutils.segmentation.SegmentationImage.make_source_mask)

cat = SourceCatalog(data, segm, convolved_data=convolved_data) # construct a catalogue from the segmentation map

flux = cat.kron_flux #to access flux
idx = np.where(flux == np.nanmax(flux)) #index for where flux is largest
gal_seg_idx = cat.labels[idx] #corresponding segmentation index


segm.remove_labels(labels=gal_seg_idx[0])
mask = segm.make_source_mask()

plt.imshow(mask)
plt.savefig(data_path + "mge_legacy/mask.png", bbox_inches = "tight")
plt.close()

background_subtracted_image[np.isnan(background_subtracted_image)] = 0  #Dealing with background and NaNs

#VALUES==========================================
#file values
img = background_subtracted_image
mask = mask

sigmaPSF = sigmaPSF #pixels
normPSF = countsPSF/np.sum(countsPSF)

#constants
scale = 0.262 #arc/pixel. From: https://www.legacysurvey.org/dr10/description/

#MGE PROCESS=====================================
#Finding x and y centers
f = find_galaxy(img, plot = True)
plt.savefig(data_path + "mge_legacy/find_galaxy.png", bbox_inches = "tight")
plt.close()

#Sectors spectrometry
s = sectors_photometry(img, f.eps, f.pa, f.xpeak, f.ypeak,
                       minlevel=0.010, plot= True, mask=~mask)
plt.savefig(data_path + "mge_legacy/sectors_photometry.png", bbox_inches = "tight")
plt.close()


# mge fit sectors
m = mge_fit_sectors(radius = s.radius, angle = s.angle, counts = s.counts, eps = f.eps,
                    ngauss=900, sigmapsf=sigmaPSF, normpsf=normPSF,
                    scale=scale, plot= True, bulge_disk=0, linear=True
                    ,outer_slope=4)

plt.figure(figsize=(8,8))  
m.plot()
plt.savefig(data_path + "mge_legacy/mge_plot.png", bbox_inches = "tight")
plt.close()


#data preparation
total_counts = m.sol[0,:] 
sigma = m.sol[1,:] #pixels
qObs = m.sol[2,:]

surf = total_counts/(2*np.pi*(sigma)**2*qObs) #counts/pixel
sigma = sigma * scale #arcseconds


#FITS FILE===================
#header
header1 = fits.Header()
header1.append(card = ('Dataset','Legacy DR10'))
header1.append(card = ('Band', 'r'))
header1.append(card = ('Galaxy', 'NGC5102'))
header1.append(card = ('scale', 0.262, "arc/pix"))
header1.append(card = ('xpeak',f.xpeak,"pixels"))
header1.append(card = ('ypeak',f.ypeak,"pixels"))
header1.append(card = ('theta', f.theta, "degrees"))
header1.append(card = ('PA', f.pa, "degrees"))
header1.append(card = ('Eps', f.eps))


im_hdu =fits.PrimaryHDU(data, header1) #store cutout image

#Sectors information table
a1 = fits.Column(name = 'angle', format = 'D', unit = 'degrees', array = s.angle)
a2 = fits.Column(name = 'radius', format ='D', unit = 'pixels', array = s.radius)
a3 = fits.Column(name = 'counts', format = 'D', unit = 'counts/pix', array = s.counts)
loc_table_1 = fits.BinTableHDU.from_columns([a1,a2,a3],name='sectors')

#PSF information table
b1 = fits.Column(name = 'Counts', format='D', unit='counts', array=total_counts)
b2 = fits.Column(name = 'sigma', format='D', unit='pixels', array=m.sol[1,:])
b3 = fits.Column(name = 'qObs', format = 'D', array = qObs)
loc_table_2 = fits.BinTableHDU.from_columns([b1,b2,b3],name='mge')


hdulist = fits.HDUList([im_hdu, loc_table_1, loc_table_2])
hdulist.writeto(data_path + '/mge_legacy/mge_LEGACY.fits', overwrite = True)

#PLOT CONTOURS ==================================
m_plot = copy.deepcopy(m)
m_plot.sol[0, :] = m.sol[0, :]
m_plot.sol[1, :] = m.sol[1, :]

# Define the region of interest around the galaxy center
n = 1000
x_center = int(f.xpeak)
y_center = int(f.ypeak)
IMG = background_subtracted_image[x_center - n:x_center + n, y_center - n:y_center + n]

# Define the new center within the cropped image
xpeak_s, ypeak_s = n, n

# Plot the contours
plt.figure(figsize=(8, 8))
mge_print_contours(IMG, f.theta, xpeak_s, ypeak_s, m_plot.sol, normpsf=normPSF, sigmapsf=sigmaPSF, minlevel=0.05)
plt.title('mge_LEGACY Contour Plot')
plt.savefig(data_path + "mge_legacy/contour_plot.png", bbox_inches = "tight")
plt.close()






# #FITS FILE===================
# #header
# header1 = fits.Header()
# header1.append(card = ('Dataset','Legacy DR10'))
# header1.append(card = ('Band', 'r'))
# header1.append(card = ('Galaxy', 'NGC5102'))
# header1.append(card = ('scale', 0.262, "arc/pix"))
# header1.append(card = ('xpeak',f.xpeak,"pixels"))
# header1.append(card = ('ypeak',f.ypeak,"pixels"))
# header1.append(card = ('theta', f.theta, "degrees"))
# header1.append(card = ('PA', f.pa, "degrees"))
# header1.append(card = ('Eps', f.eps))


# im_hdu =fits.PrimaryHDU(data, header1) #store cutout image

# #Sectors information table
# a1 = fits.Column(name = 'angle', format = 'D', unit = 'degrees', array = s.angle)
# a2 = fits.Column(name = 'radius', format ='D', unit = 'pixels', array = s.radius)
# a3 = fits.Column(name = 'counts', format = 'D', unit = 'counts/pix', array = s.counts)
# loc_table_1 = fits.BinTableHDU.from_columns([a1,a2,a3],name='sectors')

# #PSF information table
# b1 = fits.Column(name = 'surf', format='D', unit='counts/pix', array=surf)
# b2 = fits.Column(name = 'sigma', format='D', unit='arcsec', array=sigma)
# b3 = fits.Column(name = 'qObs', format = 'D', array = qObs)
# loc_table_2 = fits.BinTableHDU.from_columns([b1,b2,b3],name='mge')


# hdulist = fits.HDUList([im_hdu, loc_table_1, loc_table_2])
# hdulist.writeto(data_path + 'fits/mge_LEGACY.fits', overwrite = True)







