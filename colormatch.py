#
# Color spaces and color matching lab
# James Tompkin
# CSCI 1290
# Brown University
#

import numpy as np
from skimage import color, io
from PIL import Image
import matplotlib.pyplot as plt

####################################################################################
# Functions
####################################################################################
#
# Use matpimglotlib to show a 3D scatter plot with subsampling of the input points
# https://matplotlib.org/3.2.1/gallery/mplot3d/scatter3d.html
#
# Input:
# - img: Input image whose intensity values (or 'positions') are the coordinates we wish to plot in a color space
# - colors: The colors associated with those intensity values for visual output to the screen.
# - n: Random number of pixels to plot
# - label_x: Labels for axis.
# - lim_x: Limits for the axis of the color space
def plotColor( positions, n, label_x, label_y, label_z, lim_x, lim_y, lim_z ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    h,w,c = positions.shape

    for i in range(0,n):
        y, x = np.random.randint(h), np.random.randint(w)
        ax.scatter( positions[y,x,0], positions[y,x,1], positions[y,x,2], marker='o', facecolors=positions[y,x,:]/255.0 )

    # e.g., for RGB, this would be 'R', 'G', 'B'
    ax.set_xlabel( label_x )
    ax.set_ylabel( label_y )
    ax.set_zlabel( label_z )

    # e.g., for uint8 RGB, this would be (0,255)
    ax.set_xlim( lim_x )
    ax.set_ylim( lim_y )
    ax.set_zlim( lim_z )

    plt.show()

# Show three images in a column
def showColorTransfer( input, colortarget, transfer ):
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.set_axis_off()
    plt.imshow( input.astype(np.uint8) )
    ax = fig.add_subplot(312)
    ax.set_axis_off()
    plt.imshow( colortarget.astype(np.uint8) )
    ax = fig.add_subplot(313)
    ax.set_axis_off()
    plt.imshow( transfer.astype(np.uint8) )
    plt.show()

### part 4

source = io.imread('source.png')
texture = io.imread('texture.png')

source = color.rgb2lab(source/255.0)
texture = color.rgb2lab(texture/255.0)

out = (source - np.mean(source, axis=(0,1))) / np.std(source, axis=(0,1))
out = out * np.std(texture, axis=(0,1)) + np.mean(texture, axis=(0,1))

out = np.clip(out, [0,-100,-100], [100,100,100])
out = color.lab2rgb(out) * 255.0

io.imsave('part4.png', out)
showColorTransfer(img, ct, out)
