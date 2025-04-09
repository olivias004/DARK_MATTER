from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load the HI data cube
file_path = "/Users/livisilcock/Documents/PROJECTS/NGC5102/fits/HI_cube.fits"
f = fits.open(file_path)
data = f[0].data[0, :, :, :]  # Extract the data cube (velocity, y, x)

# ===========================
# Convert Jy/Beam → Jy/Pixel
# ===========================
bmaj = f[0].header['BMAJ'] * u.deg  # Beam major axis
bmin = f[0].header['BMIN'] * u.deg  # Beam minor axis

# Get pixel scale in degrees
wcs = WCS(f[0].header)
pixel_scale_x = np.abs(wcs.pixel_scale_matrix[0, 0])  # Degrees per pixel
pixel_scale_y = np.abs(wcs.pixel_scale_matrix[1, 1])  # Degrees per pixel
pixel_area_deg2 = pixel_scale_x * pixel_scale_y

# Compute beam area in degrees squared
beam_area_deg2 = (np.pi * (bmaj * bmin) / (4 * np.log(2))).to(u.deg**2).value

# Compute correction factor and apply it
beam_correction_factor = beam_area_deg2 / pixel_area_deg2  # Convert Jy/beam → Jy/pixel
data /= beam_correction_factor  # Corrected data
print(f"Beam Correction Factor: {beam_correction_factor:.3f} (Jy/beam → Jy/pixel)")

# ===========================
# Apply Gaussian Filtering & Masking
# ===========================
velocity_min, velocity_max = 20, 60  # Velocity range of strong HI signal
y_min, y_max = 290, 330  # Tightly mask the Y range
x_min, x_max = 290, 330  # Tightly mask the X range

# Convert beam major/minor axis to pixels
sigma_major = (bmaj.to(u.arcsec).value / (pixel_scale_x * 3600))
sigma_minor = (bmin.to(u.arcsec).value / (pixel_scale_y * 3600))
sigma_velocity = 4  # Recommended by KinMS

# Smooth data cube before masking
smoothed_data = gaussian_filter(data, sigma=[sigma_velocity, sigma_major, sigma_minor])

# Define threshold for masking
flux_threshold = 5 * np.nanstd(smoothed_data)  # 5 times noise standard deviation

# Create mask based on smoothed data
mask = smoothed_data > flux_threshold

# Apply mask to original (not smoothed) data
masked_data = np.where(mask, data, np.nan)

# Plot the mask to visualize
plt.figure(figsize=(6, 6))
plt.imshow(np.nansum(mask, axis=0), origin='lower', cmap='gray')
plt.colorbar(label="Number of velocity channels passing threshold")
plt.title("Applied Mask After Gaussian Smoothing")
plt.show()

# ===========================
# Integrate Flux 
# ===========================
# Extract velocity channel width in km/s
channel_width_kms = abs(f[0].header['CDELT3']) / 1000  # Convert m/s to km/s

# Sum spatially per velocity channel
flux_per_channel = np.nansum(masked_data, axis=(1, 2))

# Compute the integrated flux
integrated_flux = np.nansum(flux_per_channel) * channel_width_kms  # Jy km/s
print(f"Final Integrated Flux (Jy km/s): {integrated_flux:.2e}")

# ===========================
# Compute HI Mass
# ===========================
D = 4.0 # Distance in Mpc for NGC 5102
M_HI = 2.36e5 * (D ** 2) * integrated_flux  # HI mass formula

print(f"Final Corrected HI Mass: {M_HI:.2e} M_sun")

# ===========================
# Plot Integrated Flux (Moment 0 Map)
# ===========================
moment0_map = np.nansum(masked_data, axis=0)
plt.figure(figsize=(6, 6))
plt.imshow(moment0_map, origin='lower', cmap='inferno')
plt.colorbar(label="Integrated Flux (Jy/pixel km/s)")
plt.title("Final Moment 0 Map")
plt.show()


# ===========================
# Compute the Correct Velocity Axis
# ===========================
# Systemic velocity from the Mitzkus paper
systemic_velocity = 474.5  # km/s

# Extract velocity axis directly from the FITS header
reference_velocity = f[0].header['CRVAL3'] / 1000  # Central velocity in km/s
channel_width = f[0].header['CDELT3'] / 1000  # Channel width in km/s
n_channels = data.shape[0]

# Compute the velocity axis for each channel
velocity_axis = reference_velocity + (np.arange(n_channels) - f[0].header['CRPIX3'] + 1) * channel_width
velocity_axis -= systemic_velocity  # Subtract systemic velocity to center around 0

# Print diagnostics
print(f"Velocity Axis Range After Fix: {velocity_axis.min()} km/s to {velocity_axis.max()} km/s")


# ===========================
# Compute Moment 1 (Mean Velocity)
# ===========================
moment1_numerator = np.nansum(masked_data * velocity_axis[:, None, None], axis=0)
moment1_map = moment1_numerator / moment0_map
moment1_map[np.isnan(moment0_map)] = np.nan  # Mask invalid regions

# Print diagnostics
print(f"Moment 1 Map Min: {np.nanmin(moment1_map)} km/s, Max: {np.nanmax(moment1_map)} km/s")

# ===========================
# Plot the Moment 1 Map
# ===========================
plt.figure(figsize=(6, 6))
plt.imshow(moment1_map, origin='lower', cmap='coolwarm')
plt.colorbar(label="Mean Velocity (km/s)")
plt.title("Corrected Moment 1 Map (Mean Velocity)")
plt.show()


# ===========================
# Compute Moment 2 (Velocity Dispersion)
# ===========================
moment2_numerator = np.nansum(masked_data * (velocity_axis[:, None, None] - moment1_map[None, :, :])**2, axis=0)
moment2_map = np.sqrt(moment2_numerator / moment0_map)
moment2_map[np.isnan(moment0_map)] = np.nan  # Mask empty regions

# ===========================
# Plot the Moment 2 Map
# ===========================
plt.figure(figsize=(6, 6))
plt.imshow(moment2_map, origin='lower', cmap='plasma')
plt.colorbar(label="Velocity Dispersion (km/s)")
plt.title("Moment 2 Map (Velocity Dispersion)")
plt.show()

# ===========================
# Plot the Spectrum
# ===========================
velocity_channels = np.arange(len(flux_per_channel)) + velocity_min
plt.figure(figsize=(8, 4))
plt.plot(velocity_channels, flux_per_channel)
plt.xlabel("Velocity Channel")
plt.ylabel("Flux (Jy/pixel)")
plt.title("Final HI Spectrum")
plt.show()


# ===========================
# Compute Uncertainty on Moment 0 (Integrated Intensity)
# ===========================
N_channels = np.sum(mask, axis=0)  # Number of channels included in the mask
sigma_rms = np.nanstd(data)  # RMS noise across the data cube
uncertainty_moment0 = np.sqrt(N_channels) * sigma_rms * channel_width_kms  # u_I from Equation 2

# Plot the Uncertainty on Moment 0
plt.figure(figsize=(6, 6))
plt.imshow(uncertainty_moment0, origin='lower', cmap='viridis')
plt.colorbar(label="Uncertainty on Moment 0 (Jy km/s)")
plt.title("Uncertainty on Moment 0")
plt.show()

# ===========================
# Compute Uncertainty on Moment 1 (Velocity Field)
# ===========================
delta_v_line = velocity_max - velocity_min  # Spectral line width
uncertainty_moment1 = (delta_v_line / (2 * np.sqrt(3))) * (uncertainty_moment0 / moment0_map)  # u_vel from Equation 3

# Plot the Uncertainty on Moment 1
plt.figure(figsize=(6, 6))
plt.imshow(uncertainty_moment1, origin='lower', cmap='magma', vmin=-2, vmax=7)
plt.colorbar(label="Uncertainty on Moment 1 (km/s)")
plt.title("Uncertainty on Moment 1")
plt.show()  # u_vel from Equation 3



# ===========================
# Systemic velocity derivation
# ===========================









from mgefit.find_galaxy import find_galaxy
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
import numpy as np
import matplotlib.pyplot as plt

# Clean up central hole (replace NaNs with small value or zero)
cleaned_moment0 = np.nan_to_num(moment0_map, nan=0.0)

# Inspect basic structure and centre
galaxy = find_galaxy(cleaned_moment0, plot=True)
eps = galaxy.eps
pa = galaxy.pa
xpeak = galaxy.xpeak
ypeak = galaxy.ypeak

# Create a circular mask to remove noisy outskirts (optional)
radius_mask = np.hypot(*np.ogrid[:cleaned_moment0.shape[0], :cleaned_moment0.shape[1]] - np.array([[ypeak], [xpeak]])) < 150

# You can also mask the background using intensity thresholds if preferred
minlevel = np.nanpercentile(cleaned_moment0[cleaned_moment0 > 0], 10)

s = sectors_photometry(cleaned_moment0, eps, pa, xpeak, ypeak, plot=True, minlevel=minlevel, mask=~radius_mask)











