__author__ = 'jhaux'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import scipy.ndimage as ndimage
import jimlib as jim
from PIL import Image
import os


def normalize_Intensity(image_ref, image, patch, patch_2=(0,0,0,0), norm_crit='linear'):
    ''' retruns a normalized version of image_norm to fit the intensity of image_ref'''
    x_1,x_2, y_1,y_2 = patch
    a_1,a_2, b_1,b_2 = patch_2

    mean_ref_1 = np.mean(np.asarray(image_ref, dtype='float')[y_1:y_2, x_1:x_2])
    mean_im_1  = np.mean(np.asarray(image,     dtype='float')[y_1:y_2, x_1:x_2])
    mean_ref_2 = np.mean(np.asarray(image_ref, dtype='float')[b_1:b_2, a_1:a_2])
    mean_im_2  = np.mean(np.asarray(image,     dtype='float')[b_1:b_2, a_1:a_2])

    if norm_crit == 'linear':
        norm_factor =  float(mean_ref_1) / float(mean_im_1)
        image_norm = np.asarray(image, dtype='float') * float(norm_factor)
        return image_norm

    elif norm_crit == 'offset':
        offset = float(mean_im_1) - float(mean_ref_1)
        image_off = np.asarray(image, dtype='float') - float(offset)
        return  image_off

    elif norm_crit == 'combined':
        offset = (float(mean_ref_1) * float(mean_im_2) - float(mean_ref_2) * float(mean_im_1)) \
                 / (float(mean_im_2) - float(mean_im_1))
        factor = (float(mean_ref_1) - float(mean_ref_2)) \
                 / (float(mean_im_2) - float(mean_im_1))
        image_comb = factor * image + offset
        return image_comb

def difference(image_1, image_2):
    ''' Substract two images and return the difference valued from 0 to 100'''
    dif = image_1.astype('float') - image_2.astype('float')
    dif -= dif.min()
    dif *= 100 / dif.max()
    return dif

def quotient(image_1, image_2):
    ''' Substract two images and return the difference valued from 0 to 100'''
    dif = image_1.astype('float') / image_2.astype('float')
    dif -= dif.min()
    dif *= 100 / dif.max()
    return dif

def global_difference(path_to_images, ref, nref, normpatch, normpatch_2=(0,0,0,0), darkframe='NONE', rot=False, norm_crit='linear', start=0, end=None, step=1):
    '''Take a bunch of images, substract a reference image from all of them
    and then shift and normalize (0,...,100) all pictures according to the overall min and max values as well as
    the specified patch that is supposed to stay at the same intensity throughout the whole measurement.
    Also do the same, exept that the images are divided by the reference.
    Both results are stored in the picture arrays allDifferences and allQuotients
    '''

    # store all picture names in one array
    allPictureNames = jim.listdir_nohidden(path_to_images)

    # load reference frame for substraction and normalization, if wanted rotated
    if rot:
        reference = jim.rotate_90(cv2.imread(allPictureNames[ ref]), 0)
        norm_ref  = jim.rotate_90(cv2.imread(allPictureNames[nref]), 0)
    else:
        reference = cv2.imread(allPictureNames[ ref])
        norm_ref  = cv2.imread(allPictureNames[nref])

    # if wanted substract darkcurrent
    if darkframe == 'NONE':
        pass
    else:
        reference -= darkframe

    # adjust intensity of the reference frame for substraction
    reference = normalize_Intensity(norm_ref,reference, normpatch, patch_2=normpatch_2, norm_crit=norm_crit)

    glob_min =  10000000. # initialize value for the global minimum of all pics
    glob_max = -10000000. # initialize value for the global maximum of all pics
    quot_min, quot_max = glob_min, glob_max # and again for the quotient stuff

    print "GLOBAL_DIFFERENCE: loading and adjusting images..."
    # create an array where all images can be stored in
    allDifferences = np.ndarray(shape=(len(allPictureNames), reference.shape[0], reference.shape[1], reference.shape[2]))
    allQuotients   = np.ndarray(shape=(len(allPictureNames), reference.shape[0], reference.shape[1], reference.shape[2]))
    # Take all differences and store them in the already created array
    for image_name, i in zip(allPictureNames, np.arange(len(allPictureNames))[start:end][::step]):
        # load images, if wanted rotated
        if rot:
            image = jim.rotate_90(cv2.imread(image_name), 0)
        else:
            image = cv2.imread(image_name)

        # substract darkcurrent
        if darkframe == 'NONE':
            pass
        else:
            image -= darkframe # substract darkcurrent from each picture

        # now adjust the intensity of the images!
        image = normalize_Intensity(norm_ref, image, normpatch, patch_2=normpatch_2, norm_crit=norm_crit)

        # Do the substraction
        diff = image.astype('float') - reference.astype('float')
        if diff.min() < glob_min:
            glob_min = diff.min()
        if diff.max() > glob_max:
            glob_max = diff.max()

        # store the results in the above created array to pass them back
        allDifferences[i] = diff

        # Do the division
        quot = reference.astype('float') / image.astype('float')
        if quot.min() < quot_min:
            quot_min = diff.min()
        if quot.max() > quot_max:
            quot_max = diff.max()

        # store the results in the above created array to pass them back
        allQuotients[i] = quot

    # set the smallest value to zero
    for image, i in zip(allDifferences, np.arange(allDifferences.shape[0])):
        image -= glob_min
        image *= 100/(glob_max-glob_min)
        allDifferences[i] = image

    # set the smallest value to zero
    for image, i in zip(allQuotients, np.arange(allQuotients.shape[0])):
        image -= quot_min
        image *= 100/(quot_max-quot_min)
        allQuotients[i] = image

    return allDifferences, allQuotients



def global_norm(images, patch, normref):
    '''Adjust intensities for all images of one array to the intensity of the image given by the index "normref".
    NOT USED!!! DELETE?'''
    image_ref = images[normref]
    for image, i in zip(images, np.arange(len(images))):
        images[i] = normalize_Intensity(image_ref, image, patch)
        # images[i] *= 100. / float(image.max())

    return images

def mean_image(path_to_images, rot=False):
    '''Take a bunch of images and make a mean image out of them
    NOT YET TESTED!!!'''
    images = jim.listdir_nohidden(path_to_images)

    print "MEAN_IMAGE: loading images"
    for image, i in zip(images, np.arange(len(images))):
        if rot:
            im = cv2.imread(image)
            im_rot = jim.rotate_90(im, 0)
            images[i] = im_rot
        else:
            images[i] = cv2.imread(image)

    print "MEAN_IMAGE: creating mean dark frame"
    mean = np.zeros(images[0].shape)
    for image in images:
        mean += image
    mean /= float(len(images))

    return mean

def show_as_cmap(image, title, savename='show_as_cmap', lower_border=0, upper_border=100, gauss=True, patch_view=True,
                 show_finger_bars=True,
                 patch=(0,0,100,100), intensities=(0,0), bar_col='white', bar_alph=0.6, bar_w=10):
    X,Y,Z = image.shape
    x   = np.arange(Y)  # Don't ask me, but it has to be this way. Otherwise the dimensions don't work.
    y   = np.arange(X)
    z   = np.mean(image, axis=-1) # Take mean of RGB channels. All three should be the same!

    levels = MaxNLocator(nbins=5*(upper_border-lower_border)).tick_values(lower_border, upper_border)
    cmap = plt.get_cmap('jet')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    if patch_view:
        aspect = (float(patch[1])-float(patch[0]))/(float(patch[3])-float(patch[2]))
    else:
        aspect = float(Y)/float(X)
    xdim, ydim = int(aspect*10.), 10
    # print X, Y, aspect, xdim, ydim

    fig1 = plt.figure(figsize=(xdim,ydim))
    ax = plt.subplot(1, 1, 1)
    if gauss:
        z = ndimage.gaussian_filter(z, sigma=2.0, order=0)
    im = plt.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading='gaussian')
    plt.colorbar()
    # set the limits of the plot to the limits of the data
    if show_finger_bars:
        x1, x2, y1, y2 = patch
        for val, i in zip(intensities, np.arange(len(intensities))):
            if val != 0:
                ax.add_patch(Rectangle((x1 + i-int(bar_w/2), y1),bar_w, val, color=bar_col, alpha=bar_alph))
    if patch_view:
        limits = (patch[0], patch[1], patch[3], patch[2])
        plt.axis(limits)
    else:
        plt.axis([x.min(), x.max(), y.max(), y.min()])
    plt.title(title)

    fig1.savefig(savename)
    plt.close()


def concentration_colormap(image_path, image_ref_path,
                           save_name,
                           top_cut=1, bottom_cut=0,
                           right_cut=1, left_cut=0,
                           clim_bottom=0, clim_top=1,
                           gauss=True, gauss_sigma=2, gauss_order=0,
                           invert=False,
                           timestamp=True):
    '''Makes an image in falsecolor out of the difference of two input images.'''
    image_ref  = np.asarray(Image.open(image_ref_path), dtype='float32')
    image      = np.asarray(Image.open(image_path), dtype='float32')

    # crop
    image     = image[left_cut:-right_cut,bottom_cut:-top_cut]
    image_ref = image_ref[left_cut:-right_cut,bottom_cut:-top_cut]
    # rotate
    image     = np.rot90(image,     1)
    image_ref = np.rot90(image_ref, 1)
    if invert:
        dif = jim.getConcentration(image_ref, image)
    else:
        dif = jim.getConcentration(image, image_ref)

    dif[dif < 0] = 0 # saturated difference
    # dif -= dif.min()  # we dont like negative values!
    # dif /= float(dif.max()) / 100 # make it percent

    dif_g = ndimage.gaussian_filter(dif, sigma=2.0, order=0)

    imgplot = plt.imshow(dif_g)
    imgplot.set_cmap('jet')
    if clim_top == 'auto':
        clim_top = dif.max()
    if clim_bottom == 'auto':
        clim_bottom = dif.min()
    imgplot.set_clim(clim_bottom,clim_top)
    plt.xticks(())
    plt.yticks(())
    if timestamp:
        tstamp = jim.get_timestamp(image_path)
        plt.xlabel(tstamp)

    cbar = plt.colorbar()
    cbar.set_label('Change of Intensity [%]')

    # plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    return dif.max()


# ======================================================================================================================
# ==================  some helpers  ====================================================================================
# ======================================================================================================================

def easy_cmap(image):
    plt.imshow(image, cmap='jet')
    plt.show()

def plot_patch(im, patch):
    '''helps finding the right patch you want to use in your picture!'''
    fig = plt.figure()
    a = fig.add_subplot(111)
    a.imshow(im)
    x,y = patch[0],patch[2]
    w,h = patch[1] - patch[0], patch[3] - patch[2]
    a.add_patch(Rectangle((x,y),w,h,facecolor='grey', alpha=0.4))
    plt.show()

def show_1(im_1):
    fig = plt.figure()
    a = fig.add_subplot(111)
    a.imshow(im_1)
    plt.show()

def show_2(im_1, im_2):
    fig = plt.figure()
    a = fig.add_subplot(121)
    a.imshow(im_1)
    b = fig.add_subplot(122)
    b.imshow(im_2)
    plt.show()

def show_3(im_1, im_2, im_3):
    fig = plt.figure()
    a = fig.add_subplot(131)
    a.imshow(im_1)
    b = fig.add_subplot(132)
    b.imshow(im_2)
    c = fig.add_subplot(133)
    c.imshow(im_3)
    plt.show()


# ======================================================================================================================
# ==================      main      ====================================================================================
# ======================================================================================================================

def main():
    # image_path     = '/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/630_nm/measurement_2015-02-02_14-03-19_1422884274992310.ppm'
    # image_ref_path = '/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/630_nm/measurement_2015-02-02_14-03-19_1422882250253063.ppm'
    # save_name = 'savior.png'
    image_path     = '/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2014-12-17_15-50-16/measurement_2014-12-17_15-50-16/images/630_nm/measurement_2014-12-17_15-50-16_1420202334725190.ppm'
    image_ref_path = '/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2014-12-17_15-50-16/measurement_2014-12-17_15-50-16/images/630_nm/measurement_2014-12-17_15-50-16_1418891073390205.ppm'
    save_name = 'savior.png'



    concentration_colormap(image_path=image_path, image_ref_path=image_ref_path, save_name=save_name,
                           invert=True,
                           clim_bottom=0.40, clim_top=0.63
                           )

if __name__ == '__main__':
    main()