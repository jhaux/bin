__author__ = 'jhaux'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def plot_single_timestep(all_data):
    '''input must be a one-column array!'''

    intensities = all_data[1:]
    timestep    = all_data[0]

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(intensities)
    ax.set_title(format_time(timestep))
    plt.show()

def plot_all_timesteps(all_data):

    intensities = all_data[:,1:]
    timesteps   = all_data[:, 0]
    pixels      = np.arange(len(intensities[0]))
    colors='r' #???

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    zticks = []
    for i in np.arange(len(timesteps)):
        ax.plot(xs=pixels, ys=intensities[i], zs=timesteps[i], zdir='z', c=colors)
        zticks.append(format_time(timesteps[i]))

    ax.set_title('all data')
    ax.set_zticks(timesteps[::6])
    ax.set_zticklabels(zticks[::6])

    plt.show()



def main():
    path_to_file = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/BCG_nopor_Test01/measurement_2015-02-02_14-03-19/images/intensities_630.csv'
    data = np.genfromtxt(path_to_file)
    # timesteps = [datetime.datetime.fromtimestamp(timestep).strftime('%Y-%m-%d %H:%M:%S') for timestep in data[:,0]]
    # intensities = data[:,1:]

    # plot_single_timestep(data[12])

    plot_all_timesteps(data)

    return 0

if __name__ == '__main__':
    main()