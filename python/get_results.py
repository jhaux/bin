'''
Fire this sweet script in the directory of the measurement you want to have a look at
and get a bunch of nice videos back.
python get_results.py reference_frame [startframe endframe]
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os            # getting the files
import jimlib as jim # my own set of functions
import sys           # commandline arguments and output
import time          # timer

if len(sys.argv) == 4:
	ref   = int(sys.argv[1])
	start = int(sys.argv[2])
	end   = int(sys.argv[3])
else:
	ref   = int(sys.argv[1])


# setup working directories
image_base_dir      = os.getcwd()
basename            = image_base_dir.split('/')[-1] # returns the last element from an array of folder

image_dir_no_Filter = image_base_dir + '/' + basename + '/images/no_filter/'
image_dir_630       = image_base_dir + '/' + basename + '/images/630_nm/'
image_dir_450       = image_base_dir + '/' + basename + '/images/450_nm/'
image_dir_dark      = image_base_dir + '/' + basename + '/images/dark/'

# put images into arrays
Images_no_Filter = os.listdir( image_dir_no_Filter )
Images_630       = os.listdir( image_dir_630 )
Images_450       = os.listdir( image_dir_450 )
Images_dark      = os.listdir( image_dir_dark )

# load reference frames as cv2 images
reference_no_Filter = cv2.imread( image_dir_no_Filter + Images_no_Filter[ref] )
reference_630       = cv2.imread( image_dir_630       + Images_630[ref] )
reference_450       = cv2.imread( image_dir_450       + Images_450[ref] )
reference_dark      = cv2.imread( image_dir_dark      + Images_dark[ref] )

# global Video parameters:
fps    = 25
fourcc = cv2.cv.CV_FOURCC(*'mp4v')
bar    =  200
height, width = reference_no_Filter.shape[0:2]
height = height + bar
shape  = (width, height)
font   = cv2.FONT_HERSHEY_SIMPLEX # so ugly


### VIDEO no Filter ###
print "\n### VIDEO no Filter ###"
t0 = time.clock()

out   = cv2.VideoWriter('no_Filter_grey.mov', fourcc, fps, shape, False)
color = cv2.VideoWriter('no_Filter_cmap.mov', fourcc, fps, shape, False)

for image in range(len(Images_no_Filter) - 1):
#for image in range(1500,1551):
    # get timestamp from filename
    timestamp = jim.get_timestamp(Images_no_Filter[image])

    orig = cv2.imread( image_dir_no_Filter + Images_no_Filter[image] ).astype('uint8')
    grey = jim.addColorBar(orig, bar_h=bar, color=(255,229,204))
    cv2.putText(grey,timestamp,(10, height - 40), font, 4,(0,0,0),2,cv2.CV_AA)
    out.write(grey)
 
    conc = jim.getConcentration(orig, reference_no_Filter)
    over = jim.overlay_colormap(conc, background=orig, threshold=10, cmap='HOT')
    frame = jim.addColorBar(over, bar_h=bar, color=(255,178,102))
    cv2.putText(frame,timestamp,(10, height - 40), font, 4,(255,255,255),2,cv2.CV_AA)
    color.write(frame)
    
    # now generate a nice output:
    jim.progress_bar( image, len(Images_no_Filter)-1, starttime=t0 )
            
out.release()
color.release()


### VIDEO 630  ###
print "\n### VIDEO 630 ###"
t0 = time.clock()

out   = cv2.VideoWriter('630_grey.mov', fourcc, fps, shape, False)
color = cv2.VideoWriter('630_cmap.mov', fourcc, fps, shape, False)

for image in range(len(Images_630) - 1):
#for image in range(1500,1551):
    # get timestamp from filename
    timestamp = jim.get_timestamp(Images_630[image])

    orig = cv2.imread( image_dir_630 + Images_630[image] ).astype('uint8')
    grey = jim.addColorBar(orig, bar_h=bar, color=(255,229,204))
    cv2.putText(grey,timestamp,(10, height - 40), font, 4,(0,0,0),2,cv2.CV_AA)
    out.write(grey)

    conc = jim.getConcentration(orig, reference_630)
    over = jim.overlay_colormap(conc, background=orig, threshold=10, cmap='HOT')
    frame = jim.addColorBar(over, bar_h=bar, color=(255,178,102))
    cv2.putText(frame,timestamp,(10, height - 40), font, 4,(255,255,255),2,cv2.CV_AA)
    color.write(frame)

    # now generate a nice output:
    jim.progress_bar( image, len(Images_630)-1, starttime=t0 )

out.release()
color.release()


### VIDEO 450  ###
print "\n### VIDEO 450 ###"
t0 = time.clock()

out   = cv2.VideoWriter('450_grey.mov', fourcc, fps, shape, False)
color = cv2.VideoWriter('450_cmap.mov', fourcc, fps, shape, False)

for image in range(len(Images_450) - 1):
#for image in range(1500,1551):
    # get timestamp from filename
    timestamp = jim.get_timestamp(Images_450[image])

    orig = cv2.imread( image_dir_450 + Images_450[image] ).astype('uint8')
    grey = jim.addColorBar(orig, bar_h=bar, color=(255,229,204))
    cv2.putText(grey,timestamp,(10, height - 40), font, 4,(0,0,0),2,cv2.CV_AA)
    out.write(grey)

    conc = jim.getConcentration(orig, reference_450)
    over = jim.overlay_colormap(conc, background=orig, threshold=10, cmap='HOT')
    frame = jim.addColorBar(over, bar_h=bar, color=(255,178,102))
    cv2.putText(frame,timestamp,(10, height - 40), font, 4,(255,255,255),2,cv2.CV_AA)
    color.write(frame)

    # now generate a nice output:
    jim.progress_bar( image, len(Images_450)-1, starttime=t0 )

out.release()
color.release()


print "\n Finished!"
