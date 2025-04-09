#purpose: Generatign psf mges
#author: Olivia Silcock
#date:

#imports
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
from pylab import figure, cm


from astropy.convolution import convolve
from astropy.coordinates import Angle
from astropy import units as u
from astropy.io import fits
from astropy.modeling.models import Ellipse2D
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.visualization import (ZScaleInterval,MinMaxInterval,SqrtStretch,ImageNormalize)

import mgefit
from mgefit.find_galaxy import find_galaxy
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors_regularized import mge_fit_sectors_regularized
from mgefit.mge_print_contours import mge_print_contours

from photutils.datasets import make_100gaussians_image
from photutils.segmentation import (detect_sources,
                                    make_2dgaussian_kernel)
from photutils.segmentation import SourceCatalog

# load the data: -------------------------------------------------
pathname = "/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/cutout_200.4900_-36.6302.fits"
f = fits.open(pathname)
header = f[0].header # look in here for what order the filters are given in. 
data = f[0].data # the image data. 
data = data[1,:,:]
plt.imshow(data,origin='lower', vmin=0.02,vmax=2)
plt.close()
 

 #Dealing with background and NaNs -------------------------------------------------
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

mean, _, std = sigma_clipped_stats(data)
data -=mean
threshold = 3.*std
kernel = make_2dgaussian_kernel(10, size=13)
convolved_data = convolve(data, kernel)
segm = detect_sources(convolved_data, threshold, npixels=10)

#segmentation steps ---------------------------------------
from photutils.segmentation import SourceCatalog
cat = SourceCatalog(data, segm, convolved_data=convolved_data)
print(cat)

columns = ["label","max_value","kron_flux","kron_radius","maxval_index"]
tb = cat.to_table(columns=columns)

tb.sort(["kron_flux"], reverse=True)
n = 500
rows_to_remove = np.arange(n,len(tb),1)
tb.remove_rows(rows_to_remove)

print(tb)


#STAR ONE --------------------------------------
position = (2502,2792) 
cutout = Cutout2D(data, position, (50,50))
plt.imshow(cutout.data,origin='lower')
plt.close()

#finding galaxy
f_star = find_galaxy(cutout.data,fraction=0.05,plot=False)


#sectors photometry
s_star = sectors_photometry(img=cutout.data, eps=0, ang=0, xc=f_star.xpeak, yc=f_star.ypeak,
                           n_sectors=6, mask=None, minlevel=0.005, plot=False)


#mge_fit
m_star = mge_fit_sectors_regularized(radius=s_star.radius, angle=s_star.angle, counts=s_star.counts, eps=0,
                    ngauss=900, sigmapsf=0, normpsf=1,
                    scale=0.262, plot= True, bulge_disk=0, linear=True
                    ,outer_slope=4,qbounds=[0.95,1]) 

plt.savefig("/Users/livisilcock/Documents/PROJECTS/NGC5102/content/psf01_sol.png", bbox_inches = "tight" ) 
plt.close()



#Generating fits file
pa = f_star.pa
scale = 0.262 #arcs/pixel

header1 = fits.Header()
header1.append(card = ('Dataset','Legacy',"arcseconds"))
header1.append(card = ('PA', pa , "P.A [degrees]"))
header1.append(card = ('scale', scale, "arcs/pixel"))
header1.append(card = ('label',996))
header1.append(card = ('xpeak',f_star.xpeak,"arcseconds"))
header1.append(card = ('ypeak',f_star.ypeak,"arcseconds"))

im_hdu =fits.PrimaryHDU(cutout.data,header1) #store cutout image

a1 = fits.Column(name='enc_counts', format='D', unit='counts', array=m_star.sol[0,:])
a2 = fits.Column(name='sigma_pix', format='D', unit='pixels', array=m_star.sol[1,:])

loc_table = fits.BinTableHDU.from_columns([a1,a2],name='PSF MGE')
hdulist = fits.HDUList([im_hdu, loc_table])
hdulist.writeto('/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/PSF_MGE_1.fits', overwrite=True)



#STAR TWO -----------------------------------------
#defining the star array
position = (451,1494) 
cutout = Cutout2D(data, position, (50,50)) #size = (50,50)
plt.imshow(cutout.data,origin='lower')
plt.close()

#finding galaxy
f_star = find_galaxy(cutout.data,fraction=0.05,plot=False)


#sectors photometry
s_star = sectors_photometry(img=cutout.data, eps=0, ang=0, xc=f_star.xpeak, yc=f_star.ypeak,
                           n_sectors=6, mask=None, minlevel=0.001, plot=False)


#mge_fit
m_star = mge_fit_sectors_regularized(radius=s_star.radius, angle=s_star.angle, counts=s_star.counts, eps=0,
                    ngauss=900, sigmapsf=0, normpsf=1,
                    scale=0.262, plot= True, bulge_disk=0, linear=True
                    ,outer_slope=4,qbounds=[0.95,1]) 

plt.savefig("/Users/livisilcock/Documents/PROJECTS/NGC5102/content/psf02_sol.png", bbox_inches = "tight" ) 
plt.close()



#Generating fits file
pa = f_star.pa
scale = 0.262 #arcs/pixel

header1 = fits.Header()
header1.append(card = ('Dataset','Legacy',"arcseconds"))
header1.append(card = ('PA', pa , "P.A [degrees]"))
header1.append(card = ('scale', scale, "arcs/pixel"))
header1.append(card = ('label',566))
header1.append(card = ('xpeak',f_star.xpeak,"arcseconds"))
header1.append(card = ('ypeak',f_star.ypeak,"arcseconds"))

im_hdu =fits.PrimaryHDU(cutout.data,header1) #store cutout image

a1 = fits.Column(name='enc_counts', format='D', unit='counts', array=m_star.sol[0,:])
a2 = fits.Column(name='sigma_pix', format='D', unit='pixels', array=m_star.sol[1,:])

loc_table = fits.BinTableHDU.from_columns([a1,a2],name='PSF MGE')
hdulist = fits.HDUList([im_hdu, loc_table])
hdulist.writeto('/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/PSF_MGE_2.fits', overwrite=True)




#STAR THREE -----------------------------------------
#defining the star array
position = (2764,2946) 
cutout = Cutout2D(data, position, (50,50)) #size = (50,50)
plt.imshow(cutout.data,origin='lower')
plt.close()

#finding galaxy
f_star = find_galaxy(cutout.data,fraction=0.05,plot=False)


#sectors photometry
s_star = sectors_photometry(img=cutout.data, eps=0, ang=0, xc=f_star.xpeak, yc=f_star.ypeak,
                           n_sectors=6, mask=None, minlevel=0.001, plot=False)


#mge_fit
m_star = mge_fit_sectors_regularized(radius=s_star.radius, angle=s_star.angle, counts=s_star.counts, eps=0,
                    ngauss=900, sigmapsf=0, normpsf=1,
                    scale=0.262, plot= True, bulge_disk=0, linear=True
                    ,outer_slope=4,qbounds=[0.95,1]) 

plt.savefig("/Users/livisilcock/Documents/PROJECTS/NGC5102/content/psf03_sol.png", bbox_inches = "tight" ) 
plt.close()



#Generating fits file
pa = f_star.pa
scale = 0.262 #arcs/pixel

header1 = fits.Header()
header1.append(card = ('Dataset','Legacy',"arcseconds"))
header1.append(card = ('PA', pa , "P.A [degrees]"))
header1.append(card = ('scale', scale, "arcs/pixel"))
header1.append(card = ('label',1061))
header1.append(card = ('xpeak',f_star.xpeak,"arcseconds"))
header1.append(card = ('ypeak',f_star.ypeak,"arcseconds"))

im_hdu =fits.PrimaryHDU(cutout.data,header1) #store cutout image

a1 = fits.Column(name='enc_counts', format='D', unit='counts', array=m_star.sol[0,:])
a2 = fits.Column(name='sigma_pix', format='D', unit='pixels', array=m_star.sol[1,:])

loc_table = fits.BinTableHDU.from_columns([a1,a2],name='PSF MGE')
hdulist = fits.HDUList([im_hdu, loc_table])
hdulist.writeto('/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/PSF_MGE_3.fits', overwrite=True)






#STAR FOUR -----------------------------------------
#defining the star array
position = (150,1014) 
cutout = Cutout2D(data, position, (50,50)) #size = (50,50)
plt.imshow(cutout.data,origin='lower')
plt.close()

#finding galaxy
f_star = find_galaxy(cutout.data,fraction=0.05,plot=False)


#sectors photometry
s_star = sectors_photometry(img=cutout.data, eps=0, ang=0, xc=f_star.xpeak, yc=f_star.ypeak,
                           n_sectors=6, mask=None, minlevel=0.0001, plot=False)


#mge_fit
m_star = mge_fit_sectors_regularized(radius=s_star.radius, angle=s_star.angle, counts=s_star.counts, eps=0,
                    ngauss=900, sigmapsf=0, normpsf=1,
                    scale=0.262, plot= True, bulge_disk=0, linear=True
                    ,outer_slope=4,qbounds=[0.95,1]) 

plt.savefig("/Users/livisilcock/Documents/PROJECTS/NGC5102/content/psf04_sol.png", bbox_inches = "tight" ) 
plt.close()



#Generating fits file
pa = f_star.pa
scale = 0.262 #arcs/pixel

header1 = fits.Header()
header1.append(card = ('Dataset','Legacy',"arcseconds"))
header1.append(card = ('PA', pa , "P.A [degrees]"))
header1.append(card = ('scale', scale, "arcs/pixel"))
header1.append(card = ('label',412))
header1.append(card = ('xpeak',f_star.xpeak,"arcseconds"))
header1.append(card = ('ypeak',f_star.ypeak,"arcseconds"))

im_hdu =fits.PrimaryHDU(cutout.data,header1) #store cutout image

a1 = fits.Column(name='enc_counts', format='D', unit='counts', array=m_star.sol[0,:])
a2 = fits.Column(name='sigma_pix', format='D', unit='pixels', array=m_star.sol[1,:])

loc_table = fits.BinTableHDU.from_columns([a1,a2],name='PSF MGE')
hdulist = fits.HDUList([im_hdu, loc_table])
hdulist.writeto('/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/PSF_MGE_4.fits', overwrite=True)





#STAR FIVE -----------------------------------------
#defining the star array
position = (602,2476) 
cutout = Cutout2D(data, position, (50,50)) #size = (50,50)
plt.imshow(cutout.data,origin='lower')
plt.close()

#finding galaxy
f_star = find_galaxy(cutout.data,fraction=0.05,plot=False)


#sectors photometry
s_star = sectors_photometry(img=cutout.data, eps=0, ang=0, xc=f_star.xpeak, yc=f_star.ypeak,
                           n_sectors=6, mask=None, minlevel=0.003, plot=False)


#mge_fit
m_star = mge_fit_sectors_regularized(radius=s_star.radius, angle=s_star.angle, counts=s_star.counts, eps=0,
                    ngauss=900, sigmapsf=0, normpsf=1,
                    scale=0.262, plot= True, bulge_disk=0, linear=True
                    ,outer_slope=4,qbounds=[0.95,1]) 

plt.savefig("/Users/livisilcock/Documents/PROJECTS/NGC5102/content/psf05_sol.png", bbox_inches = "tight" ) 
plt.close()



#Generating fits file
pa = f_star.pa
scale = 0.262 #arcs/pixel

header1 = fits.Header()
header1.append(card = ('Dataset','Legacy',"arcseconds"))
header1.append(card = ('PA', pa , "P.A [degrees]"))
header1.append(card = ('scale', scale, "arcs/pixel"))
header1.append(card = ('label',874))
header1.append(card = ('xpeak',f_star.xpeak,"arcseconds"))
header1.append(card = ('ypeak',f_star.ypeak,"arcseconds"))

im_hdu =fits.PrimaryHDU(cutout.data,header1) #store cutout image

a1 = fits.Column(name='enc_counts', format='D', unit='counts', array=m_star.sol[0,:])
a2 = fits.Column(name='sigma_pix', format='D', unit='pixels', array=m_star.sol[1,:])

loc_table = fits.BinTableHDU.from_columns([a1,a2],name='PSF MGE')
hdulist = fits.HDUList([im_hdu, loc_table])
hdulist.writeto('/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/PSF_MGE_5.fits', overwrite=True)



