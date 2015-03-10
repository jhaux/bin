'''
Fire this sweet script in the directory of the measurement you want to have a look at
and get a bunch of nice videos back.
python get_results.py reference_frame [startframe endframe]
'''

__author__ = 'jhaux'

import cv2
import numpy as np
import os            # getting the files
import jimlib as jim # my own set of functions
import image_operations as imo # some more of my functions
import sys           # commandline arguments and output
import time          # timer

def listdir_nohidden(path):
    list = [n for n in os.listdir(path) if not n.startswith(".")]
    return list


def image_Directories_and_Types(working_directory):
    '''Get the image directories and Types in an easy way. Like this you can have several types
    of images without the need of declaring everything by hand
    returns e.g. ['cwd/base/images/630_nm', 'cwd/base/images/dark', 'cwd/base/images/no_filter'], ['630_nm', 'dark', 'no_filter']'''
    basename = working_directory.split('/')[-1]

    Types = listdir_nohidden(str(working_directory + '/' + basename + '/images/'))
    Directories = [working_directory + '/' + basename + '/images/' + type for type in Types]
    # ret. e.g. ['cwd/base/images/630_nm', 'cwd/base/images/dark', 'cwd/base/images/no_filter']

    return Types, Directories

# def define_reference_path(reference, Directories):
#     ''' return a list of reference images (arrays!) as defined by the integer "reference"'''
#     References = np.zeros(len(Directories)).astype('string')
#     for image in np.arange(len(Directories)):
#         References[image] = Directories[image] + '/' + listdir_nohidden(Directories[image])[reference]
#
#     return References

def make_color_video(directory, type,
                   ref=1,
                   fps = 25, fourcc = cv2.cv.CV_FOURCC(*'mp4v'),
                   overwrite=True):
    ''' make one standarized video:
    with overlayed colormap containing information about the CO2 concentracion'''




    # Tell std out what we are doing
    infostring = "\n### COLOR VIDEO: " + type + " ###"
    print infostring

    savedir = 'color/' + type + '/'
    if os.path.isdir(savedir):
        pass
    else:
        os.mkdir(savedir)


    # make initial picture to get dimensions for the video and to get the max value for the color bar
    image0 = listdir_nohidden(directory)[100]
    max = imo.concentration_colormap(image_path=directory + '/' + image0,
                                       image_ref_path=directory + '/' + listdir_nohidden(directory)[ref],
                                       invert=False,
                                       clim_bottom='auto', clim_top='auto',
                                       save_name=savedir + image0 + '.png')
    init = cv2.imread(os.getcwd() + '/color/' + type + '/' + image0 + '.png')
    height, width = init.shape[0:2]
    shape  = (width, height)

    # Setup output files, where the video will be written in
    colorvideo = cv2.VideoWriter(str(type +'_color-map.mov'), fourcc, fps, shape, False)


    # stats for the loading bar function
    current_image    = 0
    amount_of_images = len(listdir_nohidden(directory)) - 1
    timer_start = time.clock()

    for image in listdir_nohidden(directory):
        # iterable is a list of image names!

        if overwrite:
            # generate Colormap image:
            imo.concentration_colormap(image_path=directory + '/' + image,
                                       image_ref_path=directory + '/' + listdir_nohidden(directory)[ref],
                                       invert=False,
                                       clim_bottom=0, clim_top=max,
                                       save_name=savedir + image + '.png'
                                      )


        else:
            if os.path.exists(os.getcwd() + '/color/' + type + '/' + image + '.png'):
                pass
            else:
                # generate Colormap image:
                imo.concentration_colormap(image_path=directory + '/' + image,
                                           image_ref_path=directory + '/' + listdir_nohidden(directory)[ref],
                                           invert=False,
                                           clim_bottom=0, clim_top=max,
                                           save_name=savedir + image + '.png'
                                          )

        # take (generated) image and put it in the video file
        frame = cv2.imread( os.getcwd() + '/color/' + type + '/' + image + '.png' ).astype('uint8')
        colorvideo.write(frame)

        # now generate a nice loading bar:
        jim.progress_bar( current_image, amount_of_images, starttime=timer_start )
        current_image += 1


    colorvideo.release()

    print "\nfinished!"
    return 0


def simple_main():
    working_directory = os.getcwd()
    Types, Directories = image_Directories_and_Types(working_directory)
    ref = int(sys.argv[1])


    for input in zip(Directories, Types):
        make_color_video(input[0], input[1], ref)


if __name__ == '__main__':
    simple_main()
