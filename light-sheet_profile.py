# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:40:19 2021

@author: david
"""


# Modules
import math
import imageio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Gaussian model
def gaussian(x_val, bg, area, mu, sigma):
    return bg + area/(sigma*math.sqrt(2*math.pi))*np.exp(-(x_val-mu)**2/(2*sigma**2))

# Data
# stack_path = Path(r'E:\PythonScripts\Light_Sheet_Width\2.1deg_rotation.tif')
stack_path = Path(r'D:\Data\Alignment_20X\LS2_C1_20X_+0.9deg.tif')
light_sheets = imageio.volread(stack_path)
light_sheets_dim = np.shape(light_sheets)
z_step = 5

# Averaging
light_sheet_ave = np.mean(light_sheets, axis=1)
bg = np.mean(light_sheet_ave)

# Parameters
window = 50
x_val = np.array(range(2*window))
min_sig = 1e9
min_sig_id = -1
all_sig = np.empty(light_sheets_dim[0])

# Loop over light-sheets
for ii in range(light_sheets_dim[0]):
    # Index of peak
    peak_id = np.argmax(light_sheet_ave[ii][:])
    
    # Isolated peak
    peak = light_sheet_ave[ii][peak_id-window:peak_id+window]
    peak_dim = np.shape(peak)
    
    # Initial model values
    mu = sum(x_val*peak)/sum(peak)
    sigma = math.sqrt(sum((x_val-mu)**2*peak)/sum(peak))
    
    # Fit
    try:
        popt, pcov = curve_fit(gaussian, x_val, peak, p0=[bg, sum(peak), mu, sigma])
        all_sig[ii] = abs(popt[3])
    except:
        popt = (0, 0, 0, 1e9)
        all_sig[ii] = 100    
    
    # Save smallest width
    if abs(popt[3]) < min_sig:
        min_peak = peak
        min_sig = abs(popt[3])
        min_sig_id = ii
        min_popt = popt
        min_peak_id = peak_id
         
# Results
z_spacing = 1.0
pix = 6.5/20
det = 2304
print('Smallest sigma = ' + str(min_sig))
print('Slice ID = ' + str(min_sig_id))
print('Peak ID = ' + str(min_peak_id))
print('Peak position from centre [um] = ' + str((min_peak_id-det/2)*pix))
print('1/e^2 radius [px] = ' + str(2*min_sig))
print('1/e^2 radius [um] = ' + str(2*min_sig*pix))
print('FWHM [um] = ' + str(2*min_sig*pix*math.sqrt(2*math.log(2))))

# Variance of peak
variance = np.var(light_sheets[min_sig_id][:][:], axis=0)
variance = variance[min_peak_id-window:min_peak_id+window]
upper_bound = gaussian(x_val, *min_popt) + np.sqrt(variance)
lower_bound = gaussian(x_val, *min_popt) - np.sqrt(variance)

# Plot smallest light sheet width for verification
plotting = 1

if plotting:
  plt.figure()
  plt.plot(x_val*pix, min_peak, 'b', label='Mean profile')
  plt.plot(x_val*pix, gaussian(x_val, *min_popt), 'r', label='Gaussian fit')
  plt.fill_between(x_val*pix, lower_bound, upper_bound, facecolor='b',
                   alpha=0.5, label='+/-SD')
  plt.legend()
  plt.xlabel('Arbitrary Position [um]')
  plt.ylabel('Intensity [a.u.]')
  plt.grid()
  plt.show()
  
  plt.figure()
  plt.plot(2*all_sig*pix*math.sqrt(2*math.log(2)), label='FWHM(z)')
  plt.hlines(2*2*min_sig*pix*math.sqrt(2*math.log(2)), xmin = 0,
             xmax = light_sheets_dim[0], colors='tab:green',
             label='2 * min(FWHM(z))')
  plt.legend()
  plt.xlabel('Position in stack [slice ID]')
  plt.ylabel('FWHM [um]')
  plt.ylim(0, 5*2*min_sig*pix)
  plt.grid()
  plt.show()