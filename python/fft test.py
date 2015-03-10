__author__ = 'jhaux'

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import os

import image_operations as imop
import ltm_analysis as ltm
import jimlib as jim

from numpy import linspace , arange , reshape ,zeros
from scipy.fftpack import fft2 , fftfreq
from cmath import pi

# create some arbitrary data
some_data = arange(0.0 , 16384.0 , dtype = complex)

# reshape it to be a 128x128 2d grid
some_data_grid = reshape(some_data , (128 , 128) )

# assign some real spatial co-ordinates to the grid points
# first define the edge values
x_min = -250.0
x_max = 250.0
y_min = -250.0
y_max = 250

# then create some empty 2d arrays to hold the individual cell values
x_array = zeros( (128,128) , dtype = float )
y_array = zeros( (128,128) , dtype = float )

# now fill the arrays with the associated values
for row , y_value in enumerate(linspace (y_min , y_max , num = 128) ):

  for column , x_value in enumerate(linspace (x_min , x_max , num = 128) ):

    x_array[row][column] = x_value
    y_array[row][column] = y_value

# now for any row,column pair the x_array and y_array hold the spatial domain
# co-ordinates of the associated point in some_data_grid

# now use the fft to transform the data to the wavenumber domain
some_data_wavedomain = fft2(some_data_grid)

# now we can use fftfreq to give us a base for the wavenumber co-ords
# this returns [0.0 , 1.0 , 2.0 , ... , 62.0 , 63.0 , -64.0 , -63.0 , ... , -2.0 , -1.0 ]
n_value = fftfreq( 128 , (1.0 / 128.0 ) )

# now we can initialize some arrays to hold the wavenumber co-ordinates of each cell
kx_array = zeros( (128,128) , dtype = float )
ky_array = zeros( (128,128) , dtype = float )

# before we can calculate the wavenumbers we need to know the total length of the spatial
# domain data in x and y. This assumes that the spatial domain units are metres and
# will result in wavenumber domain units of radians / metre.
x_length = x_max - x_min
y_length = y_max - y_min

# now the loops to calculate the wavenumbers
for row in xrange(128):

  for column in xrange(128):

    kx_array[row][column] = ( 2.0 * pi * n_value[column] ) / x_length
    ky_array[row][column] = ( 2.0 * pi * n_value[row] ) / y_length

# now for any row,column pair kx_array , and ky_array will hold the wavedomain coordinates
# of the correspoing point in some_data_wavedomain
