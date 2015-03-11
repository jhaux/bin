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

def wavelength_analysis(path_to_data, reference, norm_reference, patch, norm_patch, storepics=False):
    '''Using a given dataset this function performs an analysis of the evolution of the wavelengths occuring during
    the fingering phase of the Hele-Shaw experiment. The resulting data is written into a file.
      path_to_data: string containing the path to the folder containing the images of interest e.g. .../images
      reference: integer defining which picture is to be used as reference frame
      patch: list containing the values (x_1, x_2, y_1, y_2) that define the rectangle that is used for the analysis
    '''
    # get the names of all images
    all630s = jim.listdir_nohidden(path_to_data + '/' + '630_nm')
    all450s = jim.listdir_nohidden(path_to_data + '/' + '450_nm')

    # create darkframe
    mean_dark = imop.mean_image(path_to_data + '/' + 'dark', rot=True)

    # create difference pictures (image - reference)
    Dif_630 = imop.global_difference(path_to_data + '/' + '630_nm', ref=reference, nref=norm_reference, normpatch=norm_patch, darkframe=mean_dark, rot=True)
    Dif_450 = imop.global_difference(path_to_data + '/' + '450_nm', ref=reference, nref=norm_reference, normpatch=norm_patch, darkframe=mean_dark, rot=True)

    print 'writing data to .csv-files'
    intensities_630 = np.ndarray(shape=(len(Dif_630), patch[1] - patch[0] + 1)) # plus 1 for timestep storage
    for image, i in zip(Dif_630, np.arange(len(intensities_630))):
        intensities_630[i,0]  = jim.get_timestamp(all630s[i], human_readable=False) # timestep in unixtime
        intensities_630[i,1:] = get_raw_intensities(np.mean(image, axis=-1), patch) # waves!
    np.savetxt(path_to_data + '/' + 'intensities_630.csv', intensities_630, delimiter='\t')

    intensities_450 = np.ndarray(shape=(len(Dif_450), patch[1] - patch[0] + 1))
    for image, i in zip(Dif_450, np.arange(len(intensities_450))):
        intensities_450[i,0]  = jim.get_timestamp(all450s[i], human_readable=False)
        intensities_450[i,1:] = get_raw_intensities(np.mean(image, axis=-1), patch)
    np.savetxt(path_to_data + '/' + 'intensities_450.csv', intensities_450, delimiter='\t')

    if storepics:
        print 'storing false color images'

        if not os.path.isdir(path_to_data + '/' + 'color_630'):   # see if the directories already exist, else create them
            os.makedirs(path_to_data + '/' + 'color_630')
        if not os.path.isdir(path_to_data + '/' + 'color_450'):
            os.makedirs(path_to_data + '/' + 'color_450')

        print '=> 630nm...'
        for image, i in zip(Dif_630, np.arange(len(all630s))):
            imop.show_as_cmap(image, title=jim.get_timestamp(all630s[i]), savename=path_to_data + '/' + 'color_630' + '/' + str(i), lower_border=0, upper_border=100)
        print '=> 450nm...'
        for image, i in zip(Dif_450, np.arange(len(all450s[0]))):
            imop.show_as_cmap(image, title=jim.get_timestamp(all450s[i]), savename=path_to_data + '/' + 'color_450' + '/' + str(i))

    print '\nWavelength-analysis finished!'

    return intensities_630, intensities_450


def main():
    path_to_data = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images'
    # path_to_data = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/BCG_nopor_Test01/measurement_2015-02-02_14-03-19/images'

    patch = (643,1779,1550,2000)
    norm_patch = (700,800,1400,1500)
    reference = 0
    norm_reference = 0

    wavelength_analysis( path_to_data,
                        reference=reference, patch=patch,
                        norm_reference=norm_reference, norm_patch=norm_patch,
                        storepics=True)


if __name__ == '__main__':
    main()