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
def plotColor( positions, colors, n, label_x, label_y, label_z, lim_x, lim_y, lim_z ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    h,w,c = positions.shape

    for i in range(0,n):
        y, x = np.random.randint(h), np.random.randint(w)
        ax.scatter( positions[y,x,0], positions[y,x,1], positions[y,x,2], marker='o', facecolors=colors[y,x,:]/255.0 )

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


# You might want to write a convenience function to normalize an image.
# def normalizeI( .... ):

#     return ....


# You might want to write a convenience function to transfer color between two images.
# def colorTransfer( ... ):

#     return ....



####################################################################################
# Script
####################################################################################
#
# Today, we're going to think about three different color spaces, 
# and perform linear transforms on those color spaces to perform color transfer.
# 
# This method of color transfer was designed by Erik Reinhard, Michael Ashikhmin, Bruce Gooch, and Peter Shirley, in 2001.
# It is described in the paper: 'Color Transfer Between Images' 
# https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
#
#
# Note: Yes, it's the same person (Erik Reinhard) who designed the global tone mapping operator we use in the HDR project!


###############################
# Part 0: Look at the pictures!
###############################
# Download the paper and have a look at the figures to get a sense of what it is we wish to accomplish.


# Next, download the three images we'll be using today. These are from Figure 1 in the paper.
# - Input image: https://cs.brown.edu/courses/cs129/labs/lab_colormatching/input.png
# - Color target image: https://cs.brown.edu/courses/cs129/labs/lab_colormatching/colortarget.png
# - Reinhard output image: https://cs.brown.edu/courses/cs129/labs/lab_colormatching/reinhard-output.png


#####################
# Code:
# Fix random seed:
np.random.seed(1290)

# Load images
img = np.array(Image.open('input.png'), dtype=np.float32)
ct = np.array(Image.open('colortarget.png'), dtype=np.float32)
# Intended output
img_ct_reinhard = np.array(Image.open('reinhard-output.png'), dtype=np.float32)


###############################################
# Part 1: Visualizing the input RGB color space
###############################################
# Let's look at what data points we're dealing with.
#
# Visualize the RGB points of the input image within an RGB color cube as a scatter plot.
# Then, compare this to the scatter plot of the RGB points of the color target image.
# We've provided a function `plotColor` to draw a scatter plot.
# 
#
# Notes: 
# - Label your axes! 'R', 'G', 'B'
# - Set the correct limits for your axes! For uint8 RGB, this would be (0,255)
#
# - Each pixel is a point - that's a lot of points to draw on a scatter plot!
# - We will subsample the points for viewing using the 'n' parameter, which will randomly select 'n' points to show.
# - Plot with 1000 points first to get a sense of the shape.
# - You will need to lower the number of points for interactive control - James' laptop can do interactive rendering (usable pan-zoom) with 100 points.
#

# Things to note:
# - Within the RGB color cube, the intensity is mixed with the color.
# - The two images have different characteristics within the space - slopes, scales of the 'surfaces'.

# plotColor( ct, ct, 300, 'red', 'green', 'blue', (0,255), (0,255), (0,255) )


#########################################
# Part 2: Color matching in the RGB space
#########################################
# What if we could transfer the color characteristics between two images by mapping the statistics of their pixels within the RGB space?
# Let's translate and scale one set of pixels to have the same statistics as the other set, such that their overall color characteristics match better.
# 
# Steps:
# - First, compute the mean and standard deviation of the input image pixels.
# - Second, compute the mean and standard deviation of the color target image pixels.
# - Third, 'normalizing' or 'standardizing' the input image pixel color distribution by subtracting its mean and dividing by its standard deviation.
# - Fourth, map the input colors into the color target by multiplying by their standard deviation and adding back the mean of the color target pixels.
# 


# Notes:
#   - Do I want a mean/stddev per color channel, or across the whole color space?
#
#   - We've provided a function `showColorTransfer()` to help you see your result.
#   - Make sure to clip your output image, as values > 255 or < 0 will be highlighted by imshow(). RGB values range [0,255]
#
#   - We could scatter plot the colors again to show their match (perhaps with an overlay), but take care:
#   - The 'locations' of the points have now changed, and are in a different limit range.
#   - The colors we want to display on the screen have not - this is what the second argument in `plotColor' is for.



# Questions:
# - What happened to the output?
# - Was it successful?

out = (img - np.mean(img, axis=(0,1))) / np.std(img, axis=(0,1))
out = (out * np.std(ct, axis=(0,1))) + np.mean(ct, axis=(0,1))
out = np.clip(out, 0, 255)

showColorTransfer(img, ct, out)

io.imsave('part2.png', out)

#########################################
# Part 3: Color matching in the luminance/chrominance space
#########################################

rgb2gray_weightr = 0.2125 
rgb2gray_weightg = 0.7154
rgb2gray_weightb = 0.0721
luminance_weight = np.reshape((rgb2gray_weightr, rgb2gray_weightg, rgb2gray_weightb), (1, 1, -1))


img_luminance = np.sum(img * luminance_weight, axis=(2), keepdims=True)
img_chroma = img / img_luminance

ct_luminance = np.sum(ct * luminance_weight, axis=(2), keepdims=True)
ct_chroma = ct / ct_luminance

out_luminance = (img_luminance - np.mean(img_luminance, axis=(0,1))) / np.std(img_luminance, axis=(0,1))
out_lumimance = out_luminance * np.std(ct_luminance, axis=(0,1)) + np.mean(ct_luminance, axis=(0,1))

out_chroma = (img_chroma - np.mean(img_chroma, axis=(0,1))) / np.std(img_chroma, axis=(0,1))
out_chroma = out_chroma * np.std(ct_chroma, axis=(0,1)) + np.mean(ct_chroma, axis=(0,1))

out = out_lumimance * out_chroma
out = np.clip(out, 0, 255)

showColorTransfer(img, ct, out)

plotColor( out, out, 300, 'red', 'green', 'blue', (0,255), (0,255), (0,255) )
io.imsave('part3.png', out)

### part 4

img_lab = color.rgb2lab(img/255.0)
ct_lab = color.rgb2lab(ct/255.0)

out = (img_lab - np.mean(img_lab, axis=(0,1))) / np.std(img_lab, axis=(0,1))
out = out * np.std(ct_lab, axis=(0,1)) + np.mean(ct_lab, axis=(0,1))

out = np.clip(out, [0,-100,-100], [100,100,100])
out = color.lab2rgb(out) * 255.0

io.imsave('part4.png', out)
showColorTransfer(img, ct, out)
