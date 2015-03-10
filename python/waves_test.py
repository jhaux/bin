__author__ = 'jhaux'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import image_operations as imop
import ltm_analysis as ltm
import jimlib as jim


path_to_pics  = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/630_nm'
path_to_dark  = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/dark'
path_to_waves = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/'

# # ==================================================================
# # all the code needed for the real analysis:
# allPics = os.listdir(path_to_pics)
# darks   = os.listdir(path_to_dark)
#
# ref = allPics[ 0]; ref = cv2.imread(path_to_pics + '/' + ref)
# im  = allPics[10]; im  = cv2.imread(path_to_pics + '/' + im)
#
# print "loading images"
# for image, i in zip(darks, np.arange(len(darks))):
#     darks[i] = cv2.imread(path_to_dark + '/' + image)
#
# print "creating mean dark frame"
# mean = np.zeros(darks[0].shape)
# for image in darks:
#     mean = mean + image
# mean = mean / float(len(darks))
#
# rot_ref  = jim.rotate_90(ref,0)
# rot_im   = jim.rotate_90(im, 0)
# rot_dark = jim.rotate_90(mean, 0)
# # plt.imshow(rot_dark)
# # plt.show()
#
# print "darkframe correction"
# dif = imop.difference(rot_im, rot_ref)
# dif_d = dif - rot_dark
# # fig = plt.figure()
# # a =  fig.add_subplot(2,1,1)
# # plt.imshow(dif[1540:1800,650:1750,0])
# # a.set_title('Before')
# # plt.colorbar(orientation = 'horizontal')
# # a = fig.add_subplot(2,1,2)
# # plt.imshow(dif_d[1540:1800,650:1750,0])
# # a.set_title('After')
# # plt.show()
#
#
# print "wavelength analysis..."
# patch = (650, 1750, 1540, 1800)
# waves = ltm.get_raw_intensities(dif[:,:,0], patch)
# np.savetxt(path_to_waves + '/' + 'waves.csv', waves)
# # ==================================================================

waves = np.loadtxt(path_to_waves + '/' + 'waves.csv')
N = waves.shape[0]
t = np.arange(N)

# # ==================================================================
# # Poly fit code
# slope, ycross = ltm.fit_trendline(waves)
# data_poly = slope*t + ycross
#
# print "find crossings"
# crossings = ltm.find_crossings(data_poly,waves)
# # plt.plot(waves)
# # plt.plot(data_poly)
# # plt.plot(crossings,data_poly[crossings], 'o')
# # plt.show()
#
#
# est_std, est_freq, est_phase, est_mean = ltm.fit_sine(waves-data_poly)
# data_fit = est_std*np.sin(est_freq*t+est_phase) + est_mean
#
# # plt.plot(waves-data_poly)
# # plt.plot(data_fit)
# # plt.show()
# # ==================================================================


# map real spacial coordinates to the pixel data:
xmin = 0.   # [m]
xmax = 0.23 # [m]

xlength = xmax - xmin

x_array = np.linspace(xmin,xmax,num=N) # spatial domain of the fingers!
# plt.plot(x_array,waves-data_poly)


rfft  = np.fft.rfft(waves)
freq  = np.fft.rfftfreq(waves.shape[0], d=0.235/waves.shape[0]) # d: Distance of datapoints in "Space-space"

# kx_array = np.zeros(freq.shape) # wavedomain coordinates
# for i in np.arange(freq.shape[0]):
#     kx_array[i] = (2.0 * np.pi * freq[i]) / xlength # http://stackoverflow.com/questions/7161417/how-to-calculate-wavenumber-domain-coordinates-from-a-2d-fft
#     # kx_array[i] = freq[i] / xlength

rfft_clean = np.empty_like(rfft)
rfft_clean[:] = rfft
rfft_handle = 25000
rfft_clean[(abs(rfft) < rfft_handle)] = 0  # filter all unwanted frequencies

waves_clean   = np.fft.irfft(rfft_clean)  # inverse fft returns cleaned wave
x_array_clean = np.linspace(xmin, xmax, num=waves_clean.shape[0])

fig = plt.figure()

a = fig.add_subplot(2,1,1)
# plt.plot(kx_array, abs(rfft))
# plt.plot(kx_array, abs(rfft_clean), 'r--', linewidth=2)
# plt.plot(kx_array, rfft_handle * np.ones(kx_array.shape[0]), 'g')
plt.plot(freq, abs(rfft))
plt.plot(freq, abs(rfft_clean), 'r--', linewidth=2)
plt.plot(freq, rfft_handle * np.ones(freq.shape[0]), 'g')
plt.ylim(0,120000)
plt.title('np.fft.rfft("Raw Data")')
plt.xlabel('wavenumber [$\mathrm{m}^{-1}$]')
plt.ylabel('$|A(x)|$')

a = fig.add_subplot(2,1,2)
plt.plot(x_array,waves, 'b')
plt.plot(x_array_clean,waves_clean, 'r')
plt.title('bandpass $|A| < ' + str(rfft_handle) + '$')
plt.xlabel('cell-width [$\mathrm{m}$]')
plt.ylabel('Intensity')

# a = fig.add_subplot(4,1,4)
# plt.plot(x_array,waves, 'b')
# plt.plot(x_array_bandpass,waves_bandpass, 'r')
# plt.title('bandpass k == 14/19/24.5')
# plt.xlabel('cell-width [$\mathrm{m}$]')
# plt.ylabel('Intensity')

plt.tight_layout()
plt.show()

