"""
This library contains all usefull functions used for retrieving information
from image sequences obtained during my Bachelor thesis.
"""

import time
import sys
import datetime
import cv2
import numpy as np
import glob
import os

def get_timestamp( filename, human_readable=True ):
    """unixtime in microseconds -> "YYYY-mm-dd HH:MM:SS".
    If wanted also returns only the unixtime.
    
    filename: string
    human_readable: Bool
    """
    
    # get timestring start and endpoint
    start = len(filename) - 20
    end   = len(filename) - 4
    
    unixtime = int(filename[start:start+10]) + float('0.' + filename[start+11:end])
    formated_time = datetime.datetime.fromtimestamp(unixtime).strftime('%Y-%m-%d %H:%M:%S')

    if human_readable:
        return formated_time
    else:
        return unixtime


def rotate_90( image, flip_mode ):
    """Rotate image by 90 degrees.
        
    image: cv2.imread()-image
    flipmode: 0 = counterclockwise
              1 = clockwise
             -1 = mirrored clockwise
             
    returns rotated cv2.imread()-image
    """
    
    rotated_image = cv2.flip( cv2.transpose(image), flip_mode )
    
    return rotated_image


def prepare( image, cropfactors=(0,0,0,0), rot=True ):
    """Do all neccesary manipulations.
        
    image: string with image adress
    shape: list(image_height, image_width) dimensions of input image
    cropfactors: list(right_cut, left_cut, top_cut, bottom_cut) default: (0,0,0,0)    
    rot: rotate 90degrees? default: True
    
    returns roated and cropped cv.imread()-image
    """
    
    right_cut, left_cut, top_cut, bottom_cut = cropfactors
    orig_height, orig_width = image.shape[0:2]

    if type(image).__module__ == np.__name__:
        # see if it's an image (thus ndarray)
        orig = image.astype('uint8')
    elif type(image) == string:
        # see if it's a path to an image (thus string)
        orig = cv2.imread(image).astype('uint8')
    else:
        print "ERROR: expected string or numpy.ndarray but recieved ", type(image)

    if rot:
        # rotate image
        orig_rot = rotate_90(orig, 0)
    
    # crop image
    orig_crop = orig_rot[0 + top_cut:orig_width - bottom_cut,0+left_cut:orig_height-right_cut,0:3]
                
    return orig_crop

def overlay_colormap(grayscale,
                     cmap = 'JET',
                     lower_threshold = 10, upper_threshold=255,
                     background = np.zeros((1,1,1)),
                     blend = 1,
                     beauty = True,
                     blur=True, blurKernel=(3,3)
                     ):
    '''Take Grayscale image and blend falsecolor map of it
        on top of the background image.
        
        grayscale:  cv2.imread()-image
        cmap: colormap flag: 'JET'
        threshold: lower limit of concentration. int from 0-255
        background: cv2.imread()-image default: black
        blend: float between 0.0 and 1.0
        blur: make the image look nicer
        '''
    
    # check if background is default
    if background.shape == (1,1,1):
        back = np.zeros(grayscale.shape).astype('uint8')
    else:
        back = background
    
    ### blend images
    # create masks
    img2gray  = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, lower_threshold, upper_threshold, cv2.THRESH_BINARY)
    mask_inv  = cv2.bitwise_not(mask)

    grayscale = (grayscale - lower_threshold)/(upper_threshold - lower_threshold) * 255
    grayscale[ grayscale > upper_threshold] = 255
    grayscale[ grayscale < lower_threshold] = 0


    # apply chosen colormap
    if cmap == 'JET':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
    elif cmap == 'AUTUMN':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_AUTUMN)
    elif cmap == 'BONE':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_BONE)
    elif cmap == 'COOL':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_COOL)
    elif cmap == 'HOT':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_HOT)
    elif cmap == 'HOT':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_AUTUMN)
    elif cmap == 'HSV':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_HSV)
    elif cmap == 'OCEAN':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_OCEAN)
    elif cmap == 'PINK':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_PINK)
    elif cmap == 'RAINBOW':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_RAINBOW)
    elif cmap == 'SPRING':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_SPRING)
    elif cmap == 'SUMMER':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_SUMMER)
    elif cmap == 'WINTER':
        false_color_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_WINTER)
    else:
        print 'Colormap does not exist!'
        pass

    if blur == True:
        ksize=blurKernel
        blurred_mask     = cv2.blur(mask, ksize)
        blurred_mask_inv = cv2.blur(mask_inv, ksize)
        # create back and foreground images
        fin_bg = cv2.bitwise_and(back, back, mask = blurred_mask_inv)
        fin_fg = cv2.bitwise_and(false_color_map, false_color_map, mask = blurred_mask)
    else:
        # create back and foreground images
        fin_bg = cv2.bitwise_and(back, back, mask = mask_inv)
        fin_fg = cv2.bitwise_and(false_color_map, false_color_map, mask = mask)
        
    # create output image
    final  = cv2.addWeighted( fin_bg, 1, fin_fg, blend, 0)

    return final

def addColorBar( image, bar_h=200, color=(0,174,255)):
    '''Add colorbar underneath image
        
    image: cv2.imread()-image
    bar_h: int height of bar
    color: (B,G,R) tuple of int from 0-255
    '''
    
    B, G, R = color
    h, w, d = image.shape
    f_shape = (h + bar_h, w, d)
    
    frame        =     np.zeros(f_shape).astype('uint8')
    frame[:,:,0] = B * np.ones((h + bar_h, w)).astype('uint8')
    frame[:,:,1] = G * np.ones((h + bar_h, w)).astype('uint8')
    frame[:,:,2] = R * np.ones((h + bar_h, w)).astype('uint8')
    
    frame[0:h,:,:] = image
    
    return frame

def getConcentration( image, reference ):
    '''Calculate difference between two images and return it as image
    '''
    
    concentration = np.zeros(reference.shape).astype('uint8')
    cv2.cv.Sub(cv2.cv.fromarray(reference),
               cv2.cv.fromarray(image),
               cv2.cv.fromarray(concentration)
               )
               
    return concentration

def progress_bar(current_val, end_val, bar_length=20, starttime=0):
        ''' Loading Bar function after http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console (User: JoeLinux)
        '''
        percent = float(current_val) / end_val
        hashes = '#' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(hashes))
        if starttime == 0:
            sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
        else:
            if current_val == 0:
                ETA = time.gmtime(0)
            else:
                ellapsed = time.clock() - starttime
                ETA = time.gmtime( (ellapsed/current_val) * end_val - ellapsed)

            sys.stdout.write("\rPercent: [{0}] {1}%, ETA in {2}h {3}m {4}s".format(hashes + spaces, int(round(percent * 100)), ETA.tm_hour, ETA.tm_min, ETA.tm_sec))
            
            if current_val == end_val:
                ETA = time.gmtime(0)
                percent = float(1)
                hashes = '#' * int(round(percent * bar_length))
                spaces = ' ' * (bar_length - len(hashes))
                sys.stdout.write("\rPercent: [{0}] {1}%, ETA in {2}h {3}m {4}s".format(hashes + spaces, int(round(percent * 100)), ETA.tm_hour, ETA.tm_min, ETA.tm_sec))
            else:
                sys.stdout.write("\rPercent: [{0}] {1}%, ETA in {2}h {3}m {4}s".format(hashes + spaces, int(round(percent * 100)), ETA.tm_hour, ETA.tm_min, ETA.tm_sec))
        
        sys.stdout.flush()

def normalize( image, reference, patch):
    ''' Take two images and adjust their intensities to match in the area of the specified patch'''
    x_1, x_2, y_1, y_2 = patch

    # check if "image" is a path or a matrix
    # if image ==
    mean_image     = cv2.mean(    image[y_1:y_2,x_1:x_2])
    mean_reference = cv2.mean(reference[y_1:y_2,x_1:x_2])
    factor         = float(mean_image) / float(mean_reference)

    output = cv2.multiply(image, np.ones(image.shape), factor)
    return output

def listdir_nohidden(path):
        return glob.glob(os.path.join(path, '*'))