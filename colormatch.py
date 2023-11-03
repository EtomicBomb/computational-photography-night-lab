#
# Color spaces and color matching lab
# James Tompkin
# CSCI 1290
# Brown University
#

import os
import glob
import numpy as np
from skimage import color, io, util, segmentation
# from PIL import Image
import matplotlib.pyplot as plt
from numpy import ma
from skimage.draw import polygon
from scipy import sparse

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

def apply_mask(image, mask):
    masked_image = image.copy()
    for i in range(masked_image.shape[2]):
        masked_image[:, :, i] *= mask
    return masked_image

def get_mask_with_mouse(image):
    print("Please click to define the polygon points. Press 'Enter' when done.")
    ip = InteractivePolygon(image)
    if ip.polygon_completed:
        mask = np.zeros(image.shape[:2], dtype=np.bool_)
        rr, cc = polygon(ip.ys, ip.xs)
        mask[rr, cc] = True
        return mask
    else:
        return None
    
class InteractivePolygon:
    def __init__(self, image):
        self.image = image
        self.xs = []
        self.ys = []
        self.fig, self.ax = plt.subplots()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
        self.kid = self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.ax.imshow(image)
        self.polygon_completed = False
        plt.show()

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.ax:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.ax.plot(self.xs, self.ys, 'r-x')

        self.fig.canvas.draw()

    def key_press(self, event):
        print('key press', event.key)
        if event.key == 'enter':
            self.fig.canvas.mpl_disconnect(self.cid)
            self.fig.canvas.mpl_disconnect(self.kid)
            self.polygon_completed = True
            plt.close(self.fig)

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

def histogram_segment_equalize(source, texture):
    pixel_count = np.product(np.shape(source)[:2])
    segments = segmentation.felzenszwalb(source, scale=500, min_size=pixel_count//50)
    unique = np.unique(segments)
    unique = np.expand_dims(unique, (1, 2))
    segments = segments == unique
    segments = np.expand_dims(segments, 3)
    output = np.zeros_like(source)
    for segment in segments:
        source_flat = np.reshape(source, (-1, 3))[np.ravel(segment)]
        texture_flat = np.reshape(texture, (-1, 3))[np.ravel(segment)]
        out = (source - np.mean(source_flat, axis=(0))) / np.std(source_flat, axis=(0))
        out = out * np.std(texture_flat, axis=(0)) + np.mean(texture_flat, axis=(0))
        output = np.where(segment, out, output)
#         plt.imshow(output)
#         plt.show()
    return np.clip(output, 0, 1) * 255.0

def histogram_equalize(source, texture):
    source = color.rgb2lab(source)
    texture = color.rgb2lab(texture)

    out = (source - np.mean(source, axis=(0,1))) / np.std(source, axis=(0,1))
    out = out * np.std(texture, axis=(0,1)) + np.mean(texture, axis=(0,1))

    out = np.clip(out, [0,-100,-100], [100,100,100])
    out = color.lab2rgb(out) * 255.0
    return out

def histogram_equalize_with_mask(source, texture, source_mask, texture_mask):
    source_lab = color.rgb2lab(source)
    texture_lab = color.rgb2lab(texture)
    source_masked = apply_mask(source_lab, source_mask)
    print(np.min(source_masked), np.max(source_masked))
    plt.imshow((source_masked - np.min(source_masked)) / (np.max(source_masked) - np.min(source_masked)) + np.min(source_masked))
    plt.show()
    texture_masked = apply_mask(texture_lab, texture_mask)
    out = np.zeros_like(source_lab)
    for i in range(3):  # Iterate over the L, a, b channels
        source_channel = source_masked[:, :, i]
        texture_channel = texture_masked[:, :, i]
        source_values = source_channel[source_channel != 0]
        texture_values = texture_channel[texture_channel != 0]
        source_mean, source_std = np.mean(source_values), np.std(source_values)
        texture_mean, texture_std = np.mean(texture_values), np.std(texture_values)
        standardized_source = (source_channel - source_mean) / source_std
        source_matched = standardized_source * texture_std + texture_mean
        source_matched *= source_mask
        source_matched = np.clip(source_matched, -100 if i else 0, 100)
        out[:, :, i] = source_matched
    out = color.lab2rgb(out) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def segment_recolor():
    # where both in the pair have the same size
    for pair in glob.glob('equal/*'):
        print(pair)
        source, = glob.glob(os.path.join(pair, '*source*'))
        source = io.imread(source)
        source = util.img_as_float32(source)

        texture, = glob.glob(os.path.join(pair, '*texture*'))
        texture = io.imread(texture)
        texture = util.img_as_float32(texture)

        out = histogram_segment_equalize(source, texture)
        out = np.clip(out, 0, 255)

        out_path = os.path.join(pair, 'output.jpg')
        out = np.asarray(out, dtype=np.uint8)
        io.imsave(out_path, out)

def naive_recolor():
    # the images in the pair can have a different size
    for pair in glob.glob('pairs/*'):
        print(pair)
        source, = glob.glob(os.path.join(pair, '*source*'))
        source = io.imread(source)
        source = util.img_as_float32(source)

        texture, = glob.glob(os.path.join(pair, '*texture*'))
        texture = io.imread(texture)
        texture = util.img_as_float32(texture)

        out = histogram_equalize(source, texture)

        out_path = os.path.join(pair, 'output.jpg')
        out = np.asarray(out, dtype=np.uint8)
        io.imsave(out_path, out)

def polygon_recolor():
    for pair in sorted(glob.glob('pairs/*')):
        
        source, = glob.glob(os.path.join(pair, '*source*'))
        source = io.imread(source)
        source = util.img_as_float32(source)

        texture, = glob.glob(os.path.join(pair, '*texture*'))
        texture = io.imread(texture)
        texture = util.img_as_float32(texture)

        for i in range(3):
            source_mask = get_mask_with_mouse(source)
            texture_mask = get_mask_with_mouse(texture)
            colored = histogram_equalize_with_mask(source, texture, source_mask, texture_mask)
            colored = util.img_as_float32(colored) 
            out = source 
            out = np.where(colored > 0, colored, out)
            np.clip(out, 0, 1, out=out)
            source = out

        # out = histogram_equalize(source, texture)
        out_path = os.path.join(pair, 'output.jpg')
        print(out.shape)
        out = np.asarray(out * 255, dtype=np.uint8)
        # out = np.fromarray((out * 255).astype(np.uint8))
        io.imsave(out_path, out)

def manual_segment_recolor():
    # where both in the pair have the same size
    source = io.imread('manual/source.jpg')
    source = util.img_as_float32(source)
    source = color.rgb2lab(source)
    texture = io.imread('manual/texture.jpg')
    texture = util.img_as_float32(texture)
    texture = color.rgb2lab(texture)
    source_masks = io.imread('manual/source_masks.png')
    source_masks = util.img_as_float32(source_masks)
    texture_masks = io.imread('manual/texture_masks.png')
    texture_masks = util.img_as_float32(texture_masks)

    colors, counts = np.unique(np.reshape(source_masks, (-1, 3)), axis=0, return_counts=True)
    colors = colors[counts > 500]

    output = (source - np.mean(source, axis=(0,1))) / np.std(source, axis=(0,1))
    output = output * np.std(texture, axis=(0,1)) + np.mean(texture, axis=(0,1))
    baseline = output

    for color_mask in colors:
        source_mask = np.all(source_masks == color_mask, axis=2, keepdims=True)
        texture_mask = np.all(texture_masks == color_mask, axis=2, keepdims=True)
        source_flat = np.reshape(source, (-1, 3))[np.ravel(source_mask)]
        texture_flat = np.reshape(texture, (-1, 3))[np.ravel(texture_mask)]
        out = (source - np.mean(source_flat, axis=(0))) / np.std(source_flat, axis=(0))
        out = out * np.std(texture_flat, axis=(0)) + np.mean(texture_flat, axis=(0))
        output = np.where(np.atleast_3d(source_mask), out, output)
    output = np.clip(color.lab2rgb(output), 0, 1)

    io.imsave('manual/naive-global-correction.jpg', (color.lab2rgb(baseline)))
    io.imsave('manual/output.jpg', output)
    return output

def main():
#     manual_segment_recolor()
#     segment_recolor()
#     naive_recolor()
#     polygon_recolor()

if __name__ == '__main__':
    main()
