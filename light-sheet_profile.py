# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:40:19 2021

@author: david

Description: this script is meant to be used with the SIMVIEW 5 light-sheet
microscope. It accepts a multipage TIFF Z-stack acquired with a cropped sensor
(640 x 1664 along the X-direction, this is the direction orthogonal to the
 light-sheet propagation direction). I use a cropped sensor because the
alignment mirror often has defects, which I try to avoid to have a better
reading of the light-sheet width. One could use a full sensor, but might
need to use the binary format for that.
"""


# Modules
import math
import imageio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


# Function definitions for courve-fitting
# 1. 1D Gaussian model to extract the sheet width or thickness
def gaussian(y_val, bg, area, mu, sigma):
    # Exponent coefficient
    AA = area / ( sigma * math.sqrt( 2 * math.pi ))
    
    # From https://en.wikipedia.org/wiki/Gaussian_function, 2nd equation
    # with an additional constant background term
    return bg + AA * np.exp( -( y_val - mu )**2 / ( 2 * sigma**2 ) )

# 2. Gaussian beam propagation
#    z_peak is the waist position in the array, which is supposed to be around
#    the centre
def gaussian_width(z_val, w_0, z_peak):
    # Rayleigh range calculation (CHANGE THOSE CONSTANTS ACCORDINGLY) <<<<<<<<
    refractive_index = 1.34 # SEAWATER @ 0.561 um [-] <<<<<<<<<<<<<<<<<<<<<<<<
    wavelength = 0.561 # [um] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # Rayleight range [um]
    z_R = math.pi * w_0**2 * refractive_index / wavelength
    
    # From https://en.wikipedia.org/wiki/Gaussian_beam, "Evolving beam width"
    w_val = w_0 * np.sqrt( 1 + ( ( z_val - z_peak ) / z_R )**2 )
    return w_val * math.sqrt( 2 * math.log( 2 ) ) # Transform into FWHM [um]


# Read Z-stack
stack_path = Path(r'E:\SIMVIEW 5 Data\20210602_20X_Zeiss_alignment\LS1_C1_20X_+1.5deg.tif')
sheet_width = imageio.volread(stack_path)

# Sheet profile parameters
# Stack dimensions
sheet_width_dim = np.shape(sheet_width)

# Step size along Z
z_spacing = 1.0 # [um] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Detection magnification
magnification = 20 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Pixel size in sample plane
pixel_size = 6.5 / magnification

# Detector size along the light-sheet direction of propagation (Z)
det = 2304

# Averaging along X (orthogonal to the light-sheet direction of propagation
# and in the detector plane) to obtain an average estimate for the light-
# sheet width
sheet_width_avg = np.mean(sheet_width, axis=1)

# Use the min light-sheet width intensity as a starting point for estimating
# the background in the curve fit
bg = np.amin(sheet_width_avg)

# Window size to extract each light-sheet as a function of Z
window = 50 # [pixels]


# Initialization
# Smallest Gaussian standard deviation (sigma) 
min_sig = 1e9 # initialization with a dummy high value
# Position of the Gaussian along Z
min_sig_id = -1 # initialization with a dummy value

# Array of all standard deviations along Z
all_sig = np.empty(sheet_width_dim[0])

# Sub-axis along the window (along Y, in the sensor plane along the light-
# sheet propagation direction)
y_val = np.array(range(2*window))

# Index along the Z-stack
z_val = np.array(range(sheet_width_dim[0]))

# Loop over the light-sheets along Z
for ii in range(sheet_width_dim[0]):
    # Find index of maximum (where the sheet is)
    peak_id = np.argmax(sheet_width_avg[ii][:])
    
    # Isolate a small window around the peak for fitting
    peak = sheet_width_avg[ii][peak_id-window:peak_id+window]

    # Starting fit values
    # Average value
    mu = sum( y_val * peak ) / sum( peak )
    # Standard deviation
    sigma = math.sqrt( sum( ( y_val - mu)**2 * peak ) / sum( peak ) )
    
    # Try a fit with the Gaussian defined in 1. above
    try:
        popt, pcov = curve_fit(gaussian, y_val, peak,
                               p0=[bg, sum(peak), mu, sigma])
        # Save the size of each sheet as its standard deviation along Z
        all_sig[ii] = abs(popt[3])
    except:
        # Otherwise, give parameters a dummy value
        popt = (0, 0, 0, 1e9)
        # and save an arbitrarily large standard deviation (we expect a light-
        # sheet waist of 1.7 um at 1/e^2, which is at 2 * sigma, if the pixel
        # size is about 0.4 um, it is about 4 pixels and 100 should be a big
        # enough value to detect the error in the fit)
        all_sig[ii] = 100    
    
    # Save smallest standard deviation and index along Z
    if abs(popt[3]) < min_sig:
        # Actual profile at the waist in the window
        min_peak = peak
        
        # Standard deviation at the waist
        min_sig = abs(popt[3])
        
        # Position of the waist along Z in index
        min_sig_id = ii
        
        # Fit parameter values for the smallest standard deviation along Z
        min_popt = popt
        
        # Position of the waist along Y on the sensor in index
        min_peak_id = peak_id


# Smooth light-sheet profile along Z (maybe it should be done on the raw data
# before running the pipeline which finds the smallest sigma to be less
# sensitive to noise) in view of fitting a Gaussian propagation model
all_sig_smooth = gaussian_filter1d(all_sig, 3)


# Fit profile (in progress)
profile_window = 10
profile_range = np.array(range(2*profile_window))
profile_raw = 2 * all_sig[min_sig_id-profile_window:min_sig_id+profile_window]
profile_raw = profile_raw * pixel_size * math.sqrt( 2 * math.log( 2 ) )

try:
    popt, pcov = curve_fit(gaussian_width, profile_range, profile_raw,
                            p0=[2*min_sig*pixel_size, profile_window])
    
    fit_fwhm = np.empty(sheet_width_dim[0])
    jj = 0
    for ii in z_val:
        fit_fwhm[jj] = gaussian_width(ii, popt[0]*math.sqrt(2*math.log(2)),
                                      round(popt[1])-profile_window+min_sig_id)
        jj += 1
except:
    print('Failed to fit the Gaussian profile')


# Results
print('Results at waist (smallest sigma):')
print('Sigma [pix] = ' + str(min_sig))
print('Slice ID (along Z-stack) = ' + str(min_sig_id))
print('Sheet position ID (along Y on sensor) = ' + str(min_peak_id))
print('Sheet position from centre [um] = '
      + str( ( min_peak_id - det / 2 ) * pixel_size ) )
print('1/e^2 radius [px] = ' + str( 2 * min_sig ) )
print('1/e^2 radius [um] = ' + str( 2 * min_sig * pixel_size ) )
print('FWHM [um] = '
      + str( 2 * min_sig * pixel_size * math.sqrt( 2 * math.log( 2 ) ) ) )

# To display the standard deviation on the sheet profile at waist
# Take the variance along Y at the Z-index of smallest sigma (waist)
variance = np.var(sheet_width[min_sig_id][:][:], axis=0)
# Extract only around the window
variance = variance[min_peak_id-window:min_peak_id+window]

# Add standard deviation (sqrt of variance) to the Gaussian fit at waist
upper_bound = gaussian(y_val, *min_popt) + np.sqrt(variance)
lower_bound = gaussian(y_val, *min_popt) - np.sqrt(variance)

# Plot light-sheet profile at waist with standard deviation
plt.figure()
plt.plot(y_val*pixel_size, min_peak, 'b', label='Mean profile')
plt.plot(y_val*pixel_size, gaussian(y_val, *min_popt), 'r', label='Gaussian fit')
plt.fill_between(y_val*pixel_size, lower_bound, upper_bound, facecolor='b',
                 alpha=0.5, label='+/-SD')
plt.legend()
plt.xlabel('Arbitrary Position [um]')
plt.ylabel('Intensity [a.u.]')
plt.grid()
plt.show()

# Plot light-sheet FWHM along Z
plt.figure()
plt.plot(2*all_sig*pixel_size*math.sqrt(2*math.log(2)), label='FWHM(z)')
plt.plot(fit_fwhm, label='Gaussian propagation model fit')
plt.hlines(2*2*min_sig*pixel_size*math.sqrt(2*math.log(2)), xmin = 0,
           xmax = sheet_width_dim[0], colors='tab:green',
           label='2 * min(FWHM(z))')
plt.legend()
plt.xlabel('Position in stack [slice ID]')
plt.ylabel('FWHM [um]')
plt.ylim(0, 5*2*min_sig*pixel_size)
plt.grid()
plt.show()