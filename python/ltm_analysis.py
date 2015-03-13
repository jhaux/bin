__author__ = 'jhaux'

import numpy as np
from scipy.optimize import leastsq
import pylab as plt
import image_operations as imop
import jimlib as jim
import os



def get_raw_intensities( image, patch):
    ''' Sum over all columns in a specified patch and put the results in an output file.
    returns an array with the results'''

    x_1, x_2, y_1, y_2 = patch  # x_i amd y_i together form the corners of a rectangle

    # take the sum of all columns (axis=0 in numpy) and store the results in one array:
    intensities = np.sum(image[y_1:y_2,x_1:x_2], axis=0)

    return intensities

def fit_sine(data):
    ''' fit a sin curve to the data created by "get_raw_intensities" and return the parameters.
    data: e.g. the intensities
    N:    Number of data points
    t:    range of data points'''
    N = data.shape[0] # number of data points
    t = np.arange(N) # set x axis data as a range between 0 and N-1 resutling in 1 point per data point

    slope, ycross = fit_trendline(data)
    data_poly = slope*t + ycross

    guess_mean  = np.mean(data)
    guess_freq  = float(len(find_crossings(data, data_poly))) / float(N)
    guess_std   = 3*np.std(data)/(2**0.5)
    guess_phase = 0

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - data
    est_std, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_freq, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    # data_fit = est_std*np.sin(t+est_phase) + est_mean

    return est_std, est_freq, est_phase, est_mean

def fit_trendline(data):
    ''' fit a straight line to the data points to substract trends caused by the backround light?'''

    N = data.shape[0]
    t = np.arange(N)

    guess_mean  = np.mean(data)
    guess_slope = (data[N-1] - data[0])/N

    fit = np.polyfit(t,data,1)

    return fit

def find_crossings(data1,data2):
    ''' Take two datasets and compare where their difference is positive and where negative.
    Everytime the sign changes we get a crossingpoint.
    Returns an array with the positions of the points after the crossing.
    from: http://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python'''

    diff  = data1 - data2
    signs = np.sign(diff)
    signs[signs==0] = -1   # in numpy: sign(0) = 0!
    zero_crossings = np.where(np.diff(signs))[0]

    return zero_crossings

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

def fclean_intensities(intensities, handle=25000, cell_width=0.23, clean_criterium='amplitude'):
    '''all_data must be a one-column array!'''

    N = intensities.shape[0]

    waves = intensities

    # map real spacial coordinates to the pixel data:
    xmin = 0.   # [m]
    xmax = cell_width # [m]
    xlength = xmax - xmin

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

    return waves_clean, x_array_clean

def finger_length(image, patch, threshold, handle=150, clean_criterium='wavelength', cell_width=0.23):
    '''Find out where there are fingers and how long they are by looking at the image-patch of interest, applying the
    finger_count function (where) and then look from down to top at those positions, where the first pixel greater
    than the threshold appears.'''

    intensities = get_raw_intensities(image, patch)
    # get rid of the noise
    intensities_clean, meters_clean = fclean_intensities(intensities, handle=handle, clean_criterium=clean_criterium, cell_width=cell_width)
    # get the positions of the fingers
    finger_positions, N_fingers = finger_count(intensities_clean)

    # make a binery image out of the patch of the image:
    x_1, x_2, y_1, y_2 = patch
    image_patch = image[y_1:y_2,x_1:x_2]
    ydim, xdim = image_patch.shape

    # walk from top to bottom to get the fingerlength. To avoid errors take the mean of nine neighbouring pixels
    # first for: walk through columns. If there is a finger take a measurement.
    for position in np.arange(4,len(finger_positions)-5):
        if finger_positions[position] == 0:
            pass
        else:
            #second for: walk through pixels in one column
            for pixel in np.arange(ydim)[::-1]: #::-1 => look at the column from down under
                if np.mean(image_patch[pixel,position-4:position+5]) > threshold:
                    finger_positions[position] = pixel + 1 # +1 because the first counts as 0
                    break

    return finger_positions

def wavelength_analysis(path_to_data, reference, norm_reference,
                        patch, norm_patch, norm_patch_2,
                        storepics=False, norm_crit='linear',
                        do630=True, do450=False,
                        do_intensities=True, do_fingers=True, threshold=60,
                        plot_fingerlength=True):
    '''Using a given dataset this function performs an analysis of the evolution of the wavelengths occuring during
    the fingering phase of the Hele-Shaw experiment. The resulting data is written into a file.
      path_to_data: string containing the path to the folder containing the images of interest e.g. .../images
      reference: integer defining which picture is to be used as reference frame
      patch: list containing the values (x_1, x_2, y_1, y_2) that define the rectangle that is used for the analysis
    '''

    if do630:
        # get the names of all images
        all630s = jim.listdir_nohidden(path_to_data + '/' + '630_nm')

        # create darkframe
        mean_dark = imop.mean_image(path_to_data + '/' + 'dark', rot=True)

        # create difference pictures (image - reference)
        Diff_630, Quot_630 = imop.global_difference(path_to_data + '/' + '630_nm', ref=reference, nref=norm_reference, normpatch=norm_patch, normpatch_2=norm_patch_2, norm_crit=norm_crit, darkframe=mean_dark, rot=True)

        print 'generating 630nm data'
        if not os.path.isdir(path_to_data + '/' + norm_crit):
            os.makedirs(path_to_data + '/' + norm_crit)

        if do_intensities:
            intensities_diff_630 = np.ndarray(shape=(len(Diff_630), patch[1] - patch[0] + 1)) # plus 1 for timestep storage
            intensities_quot_630 = np.ndarray(shape=(len(Quot_630), patch[1] - patch[0] + 1)) # patch[0] = x_1, p[1] = x_2 => x_length
            for image, i in zip(Diff_630, np.arange(len(intensities_diff_630))):
                intensities_diff_630[i,0]  = jim.get_timestamp(all630s[i], human_readable=False) # timestep in unixtime
                intensities_diff_630[i,1:] = get_raw_intensities(np.mean(image, axis=-1), patch) # waves!

            for image, i in zip(Quot_630, np.arange(len(intensities_quot_630))):
                intensities_quot_630[i,0]  = jim.get_timestamp(all630s[i], human_readable=False) # timestep in unixtime
                intensities_quot_630[i,1:] = get_raw_intensities(np.mean(image, axis=-1), patch) # waves again!!!

            print 'writing 630nm data to .csv-files'
            np.savetxt(path_to_data + '/' + norm_crit + '/' + 'intensities_diff_630.csv', intensities_diff_630, delimiter='\t')
            np.savetxt(path_to_data + '/' + norm_crit + '/' + 'intensities_quot_630.csv', intensities_quot_630, delimiter='\t')
        if do_fingers:
            fingers_diff_630 = np.zeros(shape=(len(Diff_630), patch[1] - patch[0] + 1))
            fingers_quot_630 = np.zeros(shape=(len(Quot_630), patch[1] - patch[0] + 1))
            for image, i in zip(Diff_630, np.arange(len(fingers_diff_630))):
                fingers_diff_630[i,0]  = jim.get_timestamp(all630s[i], human_readable=False) # timestep in unixtime
                fingers_diff_630[i,1:] = finger_length(np.mean(image, axis=-1), patch, threshold=threshold) # finger_length!

            for image, i in zip(Quot_630, np.arange(len(fingers_diff_630))):
                fingers_quot_630[i,0] = jim.get_timestamp(all630s[i], human_readable=False) # timestep in unixtime
                fingers_quot_630[i,1:] = finger_length(np.mean(image, axis=-1), patch, threshold=threshold) # finger_length!!!

            np.savetxt(path_to_data + '/' + norm_crit + '/' + 'fingers_diff_630.csv', fingers_diff_630, delimiter='\t')
            np.savetxt(path_to_data + '/' + norm_crit + '/' + 'fingers_quot_630.csv', fingers_quot_630, delimiter='\t')


        if storepics:
            print 'storing false color images'

            if not os.path.isdir(path_to_data + '/' + norm_crit + '/color_630'):   # see if the directories already exist, else create them
                os.makedirs(path_to_data + '/' + norm_crit + '/color_630')

            if not os.path.isdir(path_to_data + '/' + norm_crit + '/color_630/Differences'):
                os.makedirs(path_to_data + '/' + norm_crit + '/color_630/Differences')
            if not os.path.isdir(path_to_data + '/' + norm_crit + '/color_630/Quotients'):
                os.makedirs(path_to_data + '/' + norm_crit + '/color_630/Quotients')


            print '=> 630nm diff...'
            for image, i in zip(Diff_630, np.arange(len(all630s))):
                if plot_fingerlength:
                    x1, x2, y1, y2 = patch
                    xdim, ydim = x2-x1, y2-y1
                    overlay_patch = np.zeros((ydim,xdim,3))
                    for j, val in zip(np.arange(len(fingers_diff_630[i,1:])), fingers_diff_630[i,1:]):
                        for k in np.arange(val):
                            overlay_patch[k,j-3,0] = 100
                            overlay_patch[k,j-3,1] = 100
                            overlay_patch[k,j-3,2] = 100
                            overlay_patch[k,j-2,0] = 100
                            overlay_patch[k,j-2,1] = 100
                            overlay_patch[k,j-2,2] = 100
                            overlay_patch[k,j-1,0] = 100
                            overlay_patch[k,j-1,1] = 100
                            overlay_patch[k,j-1,2] = 100
                            overlay_patch[k,j,0]   = 100
                            overlay_patch[k,j,1]   = 100
                            overlay_patch[k,j,2]   = 100
                            overlay_patch[k,j+1,0] = 100
                            overlay_patch[k,j+1,1] = 100
                            overlay_patch[k,j+1,2] = 100
                            overlay_patch[k,j+2,0] = 100
                            overlay_patch[k,j+2,1] = 100
                            overlay_patch[k,j+2,2] = 100
                            overlay_patch[k,j+3,0] = 100
                            overlay_patch[k,j+3,1] = 100
                            overlay_patch[k,j+3,2] = 100

                  # add it to the picture
                    image[y1:y2,x1:x2] += overlay_patch

                imop.show_as_cmap(image, title=jim.get_timestamp(all630s[i]), savename=path_to_data + '/' + norm_crit + '/color_630/Differences/' + str(i),
                                  lower_border=50, upper_border=80, gauss=True)
            print '=> 630nm quot...'
            for image, i in zip(Quot_630, np.arange(len(all630s))):
                if plot_fingerlength:
                    x1, x2, y1, y2 = patch
                    xdim, ydim = x2-x1, y2-y1
                    overlay_patch = np.zeros((ydim,xdim,3))
                    for j, val in zip(np.arange(4,len(fingers_diff_630[i,5:-5])), fingers_diff_630[i,5:-5]):
                        for k in np.arange(val):
                            overlay_patch[k,j-3,0] = 100
                            overlay_patch[k,j-3,1] = 100
                            overlay_patch[k,j-3,2] = 100
                            overlay_patch[k,j-2,0] = 100
                            overlay_patch[k,j-2,1] = 100
                            overlay_patch[k,j-2,2] = 100
                            overlay_patch[k,j-1,0] = 100
                            overlay_patch[k,j-1,1] = 100
                            overlay_patch[k,j-1,2] = 100
                            overlay_patch[k,j,0]   = 100
                            overlay_patch[k,j,1]   = 100
                            overlay_patch[k,j,2]   = 100
                            overlay_patch[k,j+1,0] = 100
                            overlay_patch[k,j+1,1] = 100
                            overlay_patch[k,j+1,2] = 100
                            overlay_patch[k,j+2,0] = 100
                            overlay_patch[k,j+2,1] = 100
                            overlay_patch[k,j+2,2] = 100
                            overlay_patch[k,j+3,0] = 100
                            overlay_patch[k,j+3,1] = 100
                            overlay_patch[k,j+3,2] = 100

                  # add it to the picture
                    image[y1:y2,x1:x2] += overlay_patch

                imop.show_as_cmap(image, title=jim.get_timestamp(all630s[i]), savename=path_to_data + '/' + norm_crit + '/color_630/Quotients/' + str(i),
                                  lower_border=50, upper_border=80, gauss=True)

        print '\nWe are finished with the 630nm stuff!'

    if do450:
        # now do vereything again for 450nm!
        all450s = jim.listdir_nohidden(path_to_data + '/' + '450_nm')

        Diff_450, Quot_450 = imop.global_difference(path_to_data + '/' + '450_nm', ref=reference, nref=norm_reference, normpatch=norm_patch, normpatch_2=norm_patch_2, norm_crit=norm_crit, darkframe=mean_dark, rot=True)

        print 'generating 450nm data'
        intensities_diff_450 = np.ndarray(shape=(len(Diff_450), patch[1] - patch[0] + 1)) # plus 1 for timestep storage
        intensities_quot_450 = np.ndarray(shape=(len(Quot_450), patch[1] - patch[0] + 1)) # patch[0] = x_1, p[1] = x_2 => x_length
        fingers_diff_450 = np.zeros(shape=(len(Diff_450), patch[1] - patch[0] + 1))
        fingers_quot_450 = np.zeros(shape=(len(Quot_450), patch[1] - patch[0] + 1))
        for image, i in zip(Diff_450, np.arange(len(intensities_diff_450))):
            intensities_diff_450[i,0]  = jim.get_timestamp(all450s[i], human_readable=False) # timestep in unixtime
            intensities_diff_450[i,1:] = get_raw_intensities(np.mean(image, axis=-1), patch) # waves!

        for image, i in zip(Quot_450, np.arange(len(intensities_quot_450))):
            intensities_quot_450[i,0]  = jim.get_timestamp(all450s[i], human_readable=False) # timestep in unixtime
            intensities_quot_450[i,1:] = get_raw_intensities(np.mean(image, axis=-1), patch) # waves again!!!

        for intensity, i in zip(Diff_450, np.arange(len(intensities_diff_450))):
            fingers_diff_450[i,0]  = jim.get_timestamp(all450s[i], human_readable=False) # timestep in unixtime
            fingers_diff_450[i,1:] = finger_length(np.mean(image, axis=-1), patch, threshold=51) # finger_length!

        for image, i in zip(Quot_450, np.arange(len(intensities_quot_450))):
            fingers_quot_450[i,0] = jim.get_timestamp(all450s[i], human_readable=False) # timestep in unixtime
            fingers_quot_450[i,1:] = finger_length(np.mean(image, axis=-1), patch, threshold=51) # finger_length!!!

        print 'Writung 450nm data to .csv-files'
        np.savetxt(path_to_data + '/' + norm_crit + '/' + 'intensities_diff_450.csv', intensities_diff_450, delimiter='\t')
        np.savetxt(path_to_data + '/' + norm_crit + '/' + 'intensities_quot_450.csv', intensities_quot_450, delimiter='\t')
        np.savetxt(path_to_data + '/' + norm_crit + '/' + 'fingers_diff_450.csv', fingers_diff_450, delimiter='\t')
        np.savetxt(path_to_data + '/' + norm_crit + '/' + 'fingers_quot_450.csv', fingers_quot_450, delimiter='\t')

        if storepics:
            print 'storing false color images'

            if not os.path.isdir(path_to_data + '/' + norm_crit + '/color_450'):
                os.makedirs(path_to_data + '/' + norm_crit + '/color_450')

            if not os.path.isdir(path_to_data + '/' + norm_crit + '/color_450/Differences'):
                os.makedirs(path_to_data + '/' + norm_crit + '/color_450/Differences')
            if not os.path.isdir(path_to_data + '/' + norm_crit + '/color_450/Quotients'):
                os.makedirs(path_to_data + '/' + norm_crit + '/color_450/Quotients')

            print '=> 450nm diff...'
            for image, i in zip(Diff_450, np.arange(len(all450s))):
                imop.show_as_cmap(image, title=jim.get_timestamp(all450s[i]), savename=path_to_data + '/' + norm_crit + '/color_450/Differences/' + str(i), lower_border=50, upper_border=80)
            print '=> 450nm quot...'
            for image, i in zip(Quot_450, np.arange(len(all450s))):
                imop.show_as_cmap(image, title=jim.get_timestamp(all450s[i]), savename=path_to_data + '/' + norm_crit + '/color_450/Quotients/' + str(i), lower_border=50, upper_border=80)

        print '\nWow, we\'re done with the 450nm stuff'

    return 0


def main():

    # #=== used to generate data for all plots concerning the first experiment: ============================
    #
    # # path_to_data = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images'
    # path_to_data = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/BCG_nopor_Test01/measurement_2015-02-02_14-03-19/images'
    #
    # patch = (643,1779,1550,2000)
    # norm_patch   = (700,800,1400,1500)
    # norm_patch_2 = (1600,1700,300,400)
    # reference = 0
    # norm_reference = 0
    #
    # wavelength_analysis( path_to_data,
    #                     reference=reference, patch=patch,
    #                     norm_reference=norm_reference, norm_patch=norm_patch, norm_patch_2=norm_patch_2, norm_crit='offset',
    #                     storepics=True)

    #=== used to generate data for all plots concerning the second working experiment: ============================

    # path_to_data = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/'
    path_to_data = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/BCG_nopor_Test01/measurement_2015-02-02_14-03-19/images'
    path_to_data_2 = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images'
    path_to_data_3 = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images'
    path_to_data_4 = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images'

    patch = (643, 1779, 1550, 2000) # this part of the image will be analysed
    patch_data_2 = (443, 1500, 1443, 2150)
    norm_patch   = (1190, 1310,  60, 160) # there is a paper on the background
    norm_patch_2 = ( 500,  600, 300, 400) # an area that is not to bright
    reference = 0
    norm_reference = 0

    wavelength_analysis( path_to_data,
                        reference=reference, patch=patch,
                        norm_reference=norm_reference, norm_patch=norm_patch, norm_patch_2=norm_patch_2, norm_crit='linear',
                        storepics=True, do_intensities=False, do_fingers=True, threshold=56)


if __name__ == '__main__':
    main()