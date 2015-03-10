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

def make_std_video(directory, type,
                   ref=1,
                   fps = 25, fourcc = cv2.cv.CV_FOURCC(*'mp4v'),
                   b_h = 200, b_c=(255,255,255),
                   font = cv2.FONT_HERSHEY_SIMPLEX, font_color=(0,0,0)):
    ''' make two standarized videos:
    one with and one without overlayed colormap containing information about the CO2 concentracion'''
    # Check if directory contains files. If not: abort!
    if not listdir_nohidden(directory):
        print "no images of type " + type
        pass
    elif type == 'dark':
        print "skipping dark frames"
        pass
    else:
        # adjust video hight to fit image and bar
        reference = cv2.imread(directory + '/' + listdir_nohidden(directory)[ref])
        hight, width = reference.shape[0:2]
        hight = hight + b_h
        shape  = (width, hight)

        # Tell std out what we are doing
        infostring = "\n### VIDEO " + type + " ###"
        print infostring

        # Setup output files, where the video will be written in
        grayscale = cv2.VideoWriter(str(type +'_grayscale.mov'), fourcc, fps, shape, False)
        ovrld_cmp = cv2.VideoWriter(str(type +'_color-map.mov'), fourcc, fps, shape, False)

        # stats for the loading bar function
        current_image    = 0
        amount_of_images = len(listdir_nohidden(directory)) - 1
        timer_start = time.clock()
        for image in listdir_nohidden(directory):
            # iterable is a list of image names!

            # get timestamp from filename
            timestamp = jim.get_timestamp(image)

            # Grayscale video:
            orig    = cv2.imread( directory + '/' + image ).astype('uint8')
            frame_g = jim.addColorBar(orig, bar_h=b_h, color=b_c)
            cv2.putText(frame_g,timestamp,(10, hight - 40), font, 4,(0,0,0),2,cv2.CV_AA)
            grayscale.write(frame_g)

            # Colormap video:
            conc = jim.getConcentration(orig, reference)
            over = jim.overlay_colormap(conc, background=orig,
                                        lower_threshold=3, upper_threshold=50,
                                        cmap='HOT', blurKernel=(1,1))
            frame_c = jim.addColorBar(over, bar_h=b_h, color=b_c)
            cv2.putText(frame_c,timestamp,(10, hight - 40), font, 4,(255,255,255),2,cv2.CV_AA)
            ovrld_cmp.write(frame_c)

            # now generate a nice loading bar:
            jim.progress_bar( current_image, amount_of_images, starttime=timer_start )
            current_image += 1

        grayscale.release()
        ovrld_cmp.release()

        print "\nfinished!"
    return 0


def simple_main():
    working_directory = os.getcwd()
    Types, Directories = image_Directories_and_Types(working_directory)
    ref = int(sys.argv[1])
    # References = define_reference_path(int(sys.argv[1]), Directories)

    for dir, type in zip(Directories, Types):
        make_std_video(dir, type, ref)


if __name__ == '__main__':
    simple_main()
