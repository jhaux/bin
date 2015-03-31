__author__ = 'jhaux'
# -*- coding: utf8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d
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
    timestring = datetime.datetime.fromtimestamp(t2 - t1).strftime('%H:%M:%S')
    s = list(timestring)
    hour = float(s[1]) - 1
    s[1] = str(hour)[0]
    timestring = "".join(s)
    return timestring

def format_data(all_data, cell_width=0.23):
    '''Standard operations to get the wanted information out of the previously stored wavelength-data files'''
    if len(all_data.shape) > 1:
        timesteps = all_data[:,0]
        intensities = all_data[:,1:]
        pixels = np.arange(len(all_data[0,1:]))
        meters = np.linspace(0,cell_width,num=len(intensities[0]))
    else:
        timesteps = all_data[0]
        intensities = all_data[1:]
        pixels = np.arange(len(all_data[1:]))
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


def plot_wavelengthspace(all_data, step=1, start=0, end=-1, handle=25000, clean_crit='amplitude', cell_width=0.23, cmap_name='Greys', alpha=0.4, f_bars=True):
    '''TO DO: for i in it: ax.plot(fourierclean)'''

    # iterate over timesteps
    timesteps, intensities, pixels, meters = format_data(all_data[start:end][::step])

    colors = scalar_cmap(timesteps, cmap_name)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    width = 5*cell_width/len(intensities[0])
    t_0 = timesteps[0]
    for t, i in zip(timesteps, np.arange(len(timesteps))[::-1]):
        wavelengthspace, wavelengths, intensities_clean, meters_clean, wavelengthspace_clean \
            = fourier_clean(intensities[i], handle=handle, cell_width=cell_width, clean_criterium=clean_crit)

        fingers, n_fings = finger_count(intensities_clean)
        ax1.plot( meters,       intensities[i],             c=colors[i], alpha=alpha, label=dt2str(t, t_0) )
        ax2.plot( wavelengths,  abs(wavelengthspace),       c=colors[i], alpha=alpha, label=dt2str(t, t_0) )
        ax3.plot( meters_clean, intensities_clean,          c=colors[i], alpha=alpha, label=dt2str(t, t_0) )
        if f_bars:
            ax3.bar(  meters_clean - width/2, fingers * 26000,  width=width, color='k', edgecolor='')
        ax4.plot( wavelengths,  abs(wavelengthspace_clean), c=colors[i], alpha=alpha, label=dt2str(t, t_0) )
        t_0 = t
    # now make it beautiful
    pad = 1.5
    wavelengthspace, wavelengths, _, _1, _2 = fourier_clean(intensities[-1])

    ax1.set_xlabel('Distance from left cell boarder $[m]$')
    ax1.set_ylabel('Intensity (sum over column)')

    ax2.set_xlim(0,wavelengths.max() / 10)
    ax2.set_ylim(0,pad * abs(wavelengthspace[1:]).max())
    ax2.set_xlabel('Wavenumber k $[m^{-1}]$')
    ax2.set_ylabel('$|A|$')

    ax3.set_ylim(ax1.get_ylim())
    ax3.set_xlabel('Distance from left cell boarder $[m]$')
    ax3.set_ylabel('Intensity (sum over column)')

    ax4.set_xlim(0,wavelengths.max() / 10)
    ax4.set_ylim(0,pad * abs(wavelengthspace[1:]).max())
    ax4.set_xlabel('Wavenumber k $[m^{-1}]$')
    ax4.set_ylabel('$|A|$')

    # plt.legend()
    plt.suptitle('Cleaning Out the N01S3')
    plt.tight_layout()
    plt.show()


def plot_comparison(data_1, data_2, step=1, start=0, end=-1, handle=25000, clean_crit='amplitude', cell_width=0.23, cmap_name='Greys', alpha=0.4):
    # iterate over timesteps
    timesteps_1, intensities_1, pixels_1, meters_1 = format_data(data_1[start:end][::step])
    timesteps_2, intensities_2, pixels_2, meters_2 = format_data(data_2[start:end][::step])

    colors_1 = scalar_cmap(timesteps_1, cmap_name)
    colors_2 = scalar_cmap(timesteps_2, cmap_name)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    t_0 = timesteps_1[0]
    for t, i in zip(timesteps_1, np.arange(len(timesteps_1))[::-1]):
        wavelengthspace_1, wavelengths_1, intensities_clean_1, meters_clean_1, wavelengthspace_clean_1 \
            = fourier_clean(intensities_1[i], handle=handle, cell_width=cell_width, clean_criterium=clean_crit)
        wavelengthspace_2, wavelengths_2, intensities_clean_2, meters_clean_2, wavelengthspace_clean_2 \
            = fourier_clean(intensities_2[i], handle=handle, cell_width=cell_width, clean_criterium=clean_crit)

        ax1.plot( meters_clean_1, abs(intensities_clean_1),     c=colors_1[i], alpha=alpha, label=dt2str(t, t_0) )
        ax2.plot( wavelengths_1,  abs(wavelengthspace_clean_1), c=colors_1[i], alpha=alpha, label=dt2str(t, t_0) )
        ax3.plot( meters_clean_2, abs(intensities_clean_2),     c=colors_2[i], alpha=alpha, label=dt2str(t, t_0) )
        ax4.plot( wavelengths_2,  abs(wavelengthspace_clean_2), c=colors_2[i], alpha=alpha, label=dt2str(t, t_0) )
        t_0 = t

    # now make it beautiful
    pad = 1.5
    wavelengthspace_1, wavelengths_1, _, _1, _2 = fourier_clean(intensities_1[-1])
    wavelengthspace_2, wavelengths_2, _, _1, _2 = fourier_clean(intensities_2[-1])

    ax1.set_title('waves data1')
    ax2.set_title('spectrum data1')
    ax3.set_title('waves data2')
    ax4.set_title('spectrum data1')


    ax2.set_xlim(0,wavelengths_1.max() / 10)
    ax2.set_ylim(0,pad * abs(wavelengthspace_1[1:]).max())
    ax4.set_xlim(0,wavelengths_2.max() / 10)
    ax4.set_ylim(0,pad * abs(wavelengthspace_2[1:]).max())

    # plt.legend()
    plt.suptitle('Cleaning Out the N01S3')
    plt.show()

def finger_count(intensities):
    '''Take an array of intensities and look at the first and second derivative. Assuning that at every maximum there
    is a finger the first derivative needs to be zero and the second negative. Store an array of the same dimension with
    a 1 at the location of the finger and a 0 everywhere there is not. Additionally return the count of fingers.'''

    first_derivative  = np.diff(intensities)
    second_derivative = np.diff(intensities, n=2)

    result_array = np.zeros(intensities.shape)
    # result_array[ 0] = 0 # compensate, that you cannot look at all pixels
    # result_array[-1] = 0
    # result_array[-2] = 0

    n_fingers = 0
    # iterate over every pixel. -2 because len(sec_deriv) = len(intens) - 2. Start at 1, because the first pixel has no difference to the one before.
    for pixel in np.linspace(1,len(intensities)-2, num=len(intensities)-3):
        if np.diff(np.sign(first_derivative))[pixel-1] < 0 and np.sign(second_derivative)[pixel-1] == -1:
            result_array[pixel] = 1
            n_fingers += 1
        # else:
        #     result_array[pixel] = 0

    return result_array, n_fingers

def time_diffs(timesteps):
        t_0 = datetime.datetime.fromtimestamp(timesteps[0]).replace(microsecond=0)

        time_names = []
        for i in np.arange(len(timesteps)):
            t = datetime.datetime.fromtimestamp(timesteps[i]).replace(microsecond=0)
            time_names.append(t - t_0)

        return time_names

def plot_fingercount(all_data, step=1, start=0, end=-1, handle=25000, clean_crit='amplitude', cell_width=0.23, cmap_name='Greys', alpha=0.4):
    timesteps, intensities, pixels, meters = format_data(all_data[start:end][::step])

    t_0 = datetime.datetime.fromtimestamp(timesteps[0]).replace(microsecond=0)

    time_names = []
    for i in np.arange(len(timesteps)):
        t = datetime.datetime.fromtimestamp(timesteps[i]).replace(microsecond=0)
        time_names.append(t - t_0)

    colors = scalar_cmap(timesteps, cmap_name)

    n_fing = np.zeros(timesteps.shape)
    for i in np.arange(len(timesteps)):
        _,__,intensities_clean,____,_____ = fourier_clean(intensities[i], handle=handle, cell_width=cell_width, clean_criterium=clean_crit)
        _, N_f = finger_count(intensities_clean)

        n_fing[i] = N_f

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(timesteps, n_fing, '-')
    labelstep = int(5)
    ax1.set_xticks(timesteps[::labelstep])
    ax1.set_xticklabels(time_names[::labelstep])
    plt.show()

def plot_fingers(finger_data, step=1, start=0, end=-1, cmap_name='cool', alpha=0.8, bar_width=10):
    '''show a bar wherever a finger might be!'''
    timesteps, intensities, pixels, meters = format_data(finger_data[start:end][::step])
    colors = scalar_cmap(timesteps, cmap_name)
    bar = bar_width*meters.max() / len(meters)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for t, i in zip(timesteps, np.arange(len(timesteps))[::-1]):
        ax.bar(meters, intensities[i], width=bar, color=colors[i], alpha=alpha)

    ax.set_xlim(0,meters[-1])
    ax.set_ylim(0,intensities.max()*1.1)
    plt.show()

def plot_fingers_in_picture(path_to_data, data_fingers,
                            patch, norm_patch, norm_patch_2, norm_crit='linear',
                            step=1, start=0, end=None,
                            lower_border=0, upper_border=100, gauss=True,
                            save=True, show=False, savename='num',
                            reference=0, norm_reference=0):

    timesteps, intensities, pixels, meters = format_data(data_fingers[start:end][::step])

    # get the names of all images
    all630s = jim.listdir_nohidden(path_to_data + '/' + '630_nm')

    # create darkframe
    mean_dark = imop.mean_image(path_to_data + '/' + 'dark', rot=True)

    # create difference pictures (image - reference)
    Diff_630, Quot_630 = imop.global_difference(path_to_data + '/' + '630_nm',
                                                ref=reference,
                                                nref=norm_reference, normpatch=norm_patch, normpatch_2=norm_patch_2,
                                                norm_crit=norm_crit, darkframe=mean_dark, rot=True,
                                                start=start, end=end, step=step)

    print 'storing false color images with some nice bars inside!'

    if not os.path.isdir(path_to_data + '/' + norm_crit + '/color_630'):   # see if the directories already exist, else create them
        os.makedirs(path_to_data + '/' + norm_crit + '/color_630')

    if not os.path.isdir(path_to_data + '/' + norm_crit + '/color_630/Differences_with_fingers'):
        os.makedirs(path_to_data + '/' + norm_crit + '/color_630/Differences_with_fingers')
    if not os.path.isdir(path_to_data + '/' + norm_crit + '/color_630/Quotients_with_fingers'):
        os.makedirs(path_to_data + '/' + norm_crit + '/color_630/Quotients_with_fingers')

    print '=> 630nm diff...'
    for image, i in zip(Diff_630, np.arange(len(all630s))):

        # create a patch to overlay over the image
        x1, x2, y1, y2 = patch
        xdim, ydim = x2-x1, y2-y1
        overlay_patch = np.zeros((ydim,xdim,3))
        for j, val in zip(np.arange(len(intensities[i])), intensities[i]):
            for k in np.arange(val):
                overlay_patch[k,j-1,0] = 10
                overlay_patch[k,j-1,1] = 10
                overlay_patch[k,j-1,2] = 10

                overlay_patch[k,j,0] = 10
                overlay_patch[k,j,1] = 10
                overlay_patch[k,j,2] = 10

                overlay_patch[k,j+1,0] = 10
                overlay_patch[k,j+1,1] = 10
                overlay_patch[k,j+1,2] = 10

        # add it to the picture
        image[y1:y2,x1:x2] += overlay_patch

        # plot
        fig1 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.imshow(np.mean(image,axis=-1))
        title = jim.get_timestamp(all630s[i])
        plt.title(title)

        if save:
            if savename == 'num':
                fig1.savefig(savename + str(i)+'.png')
            if savename == 'time':
                fig1.savefig(savename + title + '.png')
        if show:
            plt.show()
        plt.close()

    print '\nWe are finished with the 630nm stuff!'

def plot_finger_growth(savename, data_fingers, parts=(0,None), px2cm=None, all_Files=None, xlims=(0,None), cmap='cool'):
    '''Assuming all fingers grow with the same rate, also their mean length grows with that rate.'''
    timesteps, intensities, pixels, meters = format_data(data_fingers)#[start:end][::step])
    x_min, x_max = xlims
    if all_Files == None:
        timenames = time_diffs(timesteps)
        dt = timesteps[1] - timesteps[0]
    else:
        times = [jim.get_timestamp(t, human_readable=False) for t in all_Files]
        t_0=jim.get_timestamp(all_Files[0], human_readable=False)
        timenames = [dt2str(t_0, t) for t in times]
    # dt /= 10**6
    mean_lengths = [np.array([]) for i in np.arange(len(parts)-1)]
    growths      = [np.array([]) for i in np.arange(len(parts)-1)]
    for i in np.arange(len(parts) - 1):
        start = parts[i]
        end   = parts[i+1]
        used_ints = intensities[:,start:end]
        mean_lengths[i] = [px2cm * np.mean(timestep_ints[timestep_ints > 0]) for timestep_ints in used_ints] # first iterate over all timesteps, then take the average
        growths[i]      = np.zeros(len(mean_lengths[i]))
        if i == 0:
            dt = 0
        else:
            dt = timesteps[i] - timesteps[i-1]
            growths[i][1:]  = np.diff(mean_lengths[i]) / dt

    print len(timesteps), len(mean_lengths[0]), len(growths[0])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    colors = scalar_cmap(mean_lengths, cmap_name=cmap)
    for data, grow, i in zip(mean_lengths, growths, np.arange(len(mean_lengths))):
        if px2cm == None:
            ax1.plot(timesteps, data, label=str(i), c=colors[i])
            # ax2.plot(timesteps, grow, label=str(i))
        else:
            ax1.plot(timesteps, data, label=str(i), c=colors[i])
            # ax2.plot(timesteps, grow, label=str(i))
        print data

    ax1.set_title(u'Mittlere Fingerlänge')
    ax1.set_xlabel(u'Zeit')
    if px2cm == None:
        ax1.set_ylabel(u'Mittlere Fingerlänge $[px]$')
        # ax2.set_ylabel('mean difference [pixel/s]')
    else:
        ax1.set_ylabel(u'Mittlere Fingerlänge $[cm]$')
        # ax2.set_ylabel('mean difference [cm/s]')
    ax1.set_xticklabels(timenames[::5], rotation=45)
    print jim.get_timestamp(all_Files[x_min]), jim.get_timestamp(all_Files[x_max])
    ax1.set_xlim(timesteps[x_min], timesteps[x_max])
    ax1.legend(loc=2)
    # ax2.set_title('mean differences')
    # ax2.set_xlabel('time [h:m:s]')
    # ax2.set_xticklabels(timenames, rotation=45)
    # ax2.legend()

    # plt.show()
    fig.savefig(savename, dpi=300, bbox_inches='tight')

def main():
    path_to_data = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/BCG_nopor_Test01/measurement_2015-02-02_14-03-19/images'
    # # path_to_file = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/intensities_630.csv'
    path_to_diff = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/BCG_nopor_Test01/measurement_2015-02-02_14-03-19/images/normcrit_linear/intensities_Diff_630.csv'
    path_to_quot = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/BCG_nopor_Test01/measurement_2015-02-02_14-03-19/images/normcrit_linear/intensities_Quot_630.csv'
    # path_to_diff = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/BCG_nopor_Test01/measurement_2015-02-02_14-03-19/images/intensities_Diff_630.csv'
    # path_to_quot = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/BCG_nopor_Test01/measurement_2015-02-02_14-03-19/images/intensities_Quot_630.csv'
    data_diff = np.genfromtxt(path_to_diff)
    data_quot = np.genfromtxt(path_to_quot)

    path_to_fingers = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/BCG_nopor_Test01/measurement_2015-02-02_14-03-19/images/linear/fingers_diff_630.csv'
    data_fingers = np.genfromtxt(path_to_fingers)

    # plot_wavelengthspace(data_diff, step=10, start=1, end=32, cmap_name='cool', handle=150, clean_crit='wavelength', alpha=0.5, f_bars=True)
    # plot_wavelengthspace(data_quot, step=2, start=15, end=16, cmap_name='cool', handle=1000000000000, clean_crit='wavelength', alpha=0.5, f_bars=False)
    # plot_comparison(data_diff, data_quot, step=2, start=0, end=100, cmap_name='cool', handle=150, clean_crit='wavelength', alpha=0.5)
    # plot_fingercount(data_diff, start=1, handle=150, clean_crit='wavelength')
    # plot_fingers(data_fingers, start=8, end=9, step=1)
    patch = ( 643, 1779, 1550, 2000) # this part of the image will be analysed
    np1   = (1190, 1310,   60,  160) # there is a paper on the background
    np2   = ( 500,  600,  300,  400) # an area that is not to bright
    # plot_fingers_in_picture(path_to_data, data_fingers,patch=(643, 1779, 1550, 2000), norm_patch=np1, norm_patch_2=np2,
    #                         start=1,end=32, step=10, save=True, show=True,
    #                         savename=path_to_data+'/linear/Differences_with_fingers/')

    timesteps, intensities, pixels, meters = format_data(data_fingers)#[start:end][::step])
    sep = int(len(intensities[0])/3)
    plot_finger_growth(data_fingers[5:16], parts=(0,sep,sep*2,None))

    return 0

if __name__ == '__main__':
    main()