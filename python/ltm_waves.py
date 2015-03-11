__author__ = 'jhaux'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import datetime

import jimlib as jim
import image_operations as imop

# class HS-Cell:
#     '''Base class for the Hele-Shaw experiment'''
#     # some basic properties of the cell
#     width = 23 # cm

def format_time(time):
    return datetime.datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')

def dt2str(t1, t2):
    return datetime.datetime.fromtimestamp(t2 - t1).strftime('%H:%M:%S')

def format_data(all_data, cell_width=0.23):
    '''Standard operations to get the wanted information out of the previously stored wavelength-data files'''
    if len(all_data.shape) > 1:
        timesteps = all_data[:,0]
        intensities = all_data[:,1:]
        pixels = len(all_data[0,1:])
        meters = np.linspace(0,cell_width,num=len(intensities[0]))
    else:
        timesteps = all_data[0]
        intensities = all_data[1:]
        pixels = len(all_data[1:])
        meters = np.linspace(0,cell_width,num=len(intensities))

    return timesteps, intensities, pixels, meters

def scalar_cmap(data, cmap_name='cool'):
    '''returns an array of rgba colorvalues for plotting purposes'''

    values = np.arange(len(data))
    cmap = cm = plt.get_cmap(cmap_name)
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    color_vals = [scalarMap.to_rgba(values[i]) for i in values]

    return color_vals

def plot_single_timestep(all_data):
    '''input must be a one-column array!'''

    timesteps, intensities, pixels, meters = format_data(all_data)

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(intensities)
    ax.set_title(format_time(timestep))
    plt.show()

def plot_all_timesteps3d(all_data, cmap_name):

    timesteps, intensities, pixels, meters = format_data(all_data)
    colors = scalar_cmap(timesteps, cmap_name)
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    zticks = []
    for i in np.arange(len(timesteps)):
        ax.plot(xs=pixels, ys=intensities[i], zs=timesteps[i], zdir='z', c=colors[i], alpha=1.)
        zticks.append(format_time(timesteps[i]))

    ax.set_title('all data')
    ax.set_zticks(timesteps[::6])
    ax.set_zticklabels(zticks[::6])

    plt.show()

def plot_all_timesteps2d(all_data, cmap_name):

    timesteps, intensities, pixels, meters = format_data(all_data)

    colors = scalar_cmap(timesteps, cmap_name)
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    zticks = []
    for i in np.arange(len(timesteps)):
        ax.plot(intensities[i], c=colors[i], alpha=1.)
        zticks.append(format_time(timesteps[i]))

    ax.set_title('all data')
    # ax.set_zticks(timesteps[::6])
    # ax.set_zticklabels(zticks[::6])

    plt.show()

def fourier_clean(all_data, handle=25000, cell_width=0.23, clean_criterium='amplitude'):
    '''all_data must be a one-column array!'''

    timesteps, intensities, pixels, meters = format_data(all_data)

    N = intensities.shape[0]

    waves = intensities

    # map real spacial coordinates to the pixel data:
    xmin = 0.   # [m]
    xmax = cell_width # [m]
    xlength = xmax - xmin

    x_array = meters # spatial domain of the fingers!

    # get wavelengthspace and corresponding wavelngths
    wavelengthspace  = np.fft.rfft(waves)
    wavelengths      = np.fft.rfftfreq(waves.shape[0], d=xlength/waves.shape[0]) # d: Distance of datapoints in "Space-space"

    # clean wavlengthspace
    wavelengthspace_clean = np.empty_like(wavelengthspace)
    wavelengthspace_clean[:] = wavelengthspace
    if clean_criterium == 'amplitude':
        wavelengthspace_clean[(abs(wavelengthspace) < handle)] = 0  # filter all unwanted wavelengths
    elif clean_criterium == 'wavelength':
        wavelengthspace_clean[(wavelengths > handle)] = 0

    # get cleaned version of the waves
    waves_clean   = np.fft.irfft(wavelengthspace_clean)  # inverse fft returns cleaned wave
    x_array_clean = np.linspace(xmin, xmax, num=waves_clean.shape[0])

    return wavelengthspace, wavelengths, waves_clean, x_array_clean, wavelengthspace_clean


def plot_wavelengthspace(all_data, step = 1, handle=25000, clean_crit='amplitude', cell_width=0.23, cmap_name='Greys', alpha=0.4):
    '''TO DO: for i in it: ax.plot(fourierclean)'''

    # iterate over timesteps
    timesteps, intensities, pixels, meters = format_data(all_data[::step])

    colors = scalar_cmap(timesteps, cmap_name)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    t_0 = timesteps[0]
    for t, i in zip(timesteps, np.arange(len(timesteps))[::-1]):
        wavelengthspace, wavelengths, intensities_clean, meters_clean, wavelengthspace_clean = fourier_clean(intensities[i], handle=handle, cell_width=cell_width, clean_criterium=clean_crit)
        ax1.plot( meters,       intensities[i],             c=colors[i], alpha=alpha, label=dt2str(t, t_0) )
        ax2.plot( wavelengths,  abs(wavelengthspace),       c=colors[i], alpha=alpha, label=dt2str(t, t_0) )
        ax3.plot( meters_clean, intensities_clean,          c=colors[i], alpha=alpha, label=dt2str(t, t_0) )
        ax4.plot( wavelengths,  abs(wavelengthspace_clean), c=colors[i], alpha=alpha, label=dt2str(t, t_0) )
        t_0 = t
    # now make it beautiful
    pad = 1.5
    wavelengthspace, wavelengths, _, _1, _2 = fourier_clean(intensities[-1])
    print abs(wavelengthspace).max()
    ax2.set_xlim(0,wavelengths.max() / 10)
    ax2.set_ylim(0,pad * abs(wavelengthspace[1:]).max())
    ax4.set_xlim(0,wavelengths.max() / 10)
    ax4.set_ylim(0,pad * abs(wavelengthspace[1:]).max())

    # plt.legend()
    plt.suptitle('Cleaning Out the N01S3')
    plt.show()

def main():
    path_to_file = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/intensities_630.csv'
    # path_to_file = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/BCG_nopor_Test01/measurement_2015-02-02_14-03-19/images/intensities_630.csv'
    data = np.genfromtxt(path_to_file)
    # timesteps = [datetime.datetime.fromtimestamp(timestep).strftime('%Y-%m-%d %H:%M:%S') for timestep in data[:,0]]
    # intensities = data[:,1:]

    # plot_single_timestep(data[12])
    plot_wavelengthspace(data, step=5, cmap_name='cool', handle=150, clean_crit='wavelength', alpha=0.5)
    # plot_all_timesteps3d(data, 'cool')
    # plot_all_timesteps2d(data, 'cool')



    return 0

if __name__ == '__main__':
    main()