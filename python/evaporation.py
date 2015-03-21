__author__ = 'jhaux'

import jimlib as jim
import image_operations as imop
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2


def rescale(image, low=0, top=100):
    image[image > top] = top
    image[image < low] = low

    image -= image.min()
    image *= 100/image.max()

    return image

def main():
    # Load the pictures for plotting purposes
    path_to_files = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2014-12-17_15-50-16/measurement_2014-12-17_15-50-16/images/630_nm'
    all_Files = jim.listdir_nohidden(path_to_files)

    pic_1 = 1
    pic_2 = len(all_Files) - 5
    print all_Files[pic_1]
    print all_Files[pic_2]
    yshift = 1
    xmin,xmax,ymin,ymax,cmin,cmax = 673,None,0,None,None,None
    P_1  = jim.rotate_90(cv2.imread(all_Files[pic_1]),0)[:-yshift,xmin:,:]
    P_2  = jim.rotate_90(cv2.imread(all_Files[pic_2]),0)[yshift:,xmin:,:]
    D_21 = np.mean(imop.difference(P_1,P_2),axis=-1)

    D_21 = rescale(D_21, 30, 70)

    images = (P_1, P_2, D_21)
    titles = (jim.get_timestamp(all_Files[pic_1]), jim.get_timestamp(all_Files[pic_2]), 'Difference')
    print D_21

    # plot
    fig, axes = plt.subplots(nrows=1, ncols=3)

    for ax, i in zip(axes.flat, np.arange(len(images))):
        im = ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    cbar_ax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cbar_ax, **kw)
    plt.show()
    return 0

if __name__ == '__main__':
    main()