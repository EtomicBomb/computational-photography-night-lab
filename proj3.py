"""
Project 3: Poisson - student.py
CS 1290 Computational Photography, Brown U.
Written by James Hays.
Converted to Python by Isa Milefchik.

You will be implementing Poisson blending for this assignment.

As explained on the web page, we solve for output by setting up a large
system of equations, in matrix form, which specifies the desired value or
gradient or Laplacian
(e.g., http://en.wikipedia.org/wiki/Discrete_Laplace_operator)

The comments here will walk you through a conceptually simple way to set
up the image blending, although it is not necessarily the most efficient
formulation.

We will set up a system of equations A * x = b, where A has as many rows
and columns as there are pixels in our images. Thus, a 300x200 image will
lead to A being 60000 x 60000. 'x' is our output image (a single color
channel of it) stretched out as a vector. 'b' contains two types of known
values:

    (1) For rows of A which correspond to pixels that are not under the
        mask, b will simply contain the already known value from 'target'
        and the row of A will be a row of an identity matrix. Basically,
        this is our system of equations saying "do nothing for the pixels we
        already know".
    (2) For rows of A which correspond to pixels under the mask, we will
        specify that the gradient (actually the discrete Laplacian) in the
        output should equal the gradient in 'source', according to the final
        equation in the webpage:
           4*x(i,j) - x(i-1, j) - x(i+1, j) - x(i, j-1) - x(i, j+1) =
           4*s(i,j) - s(i-1, j) - s(i+1, j) - s(i, j-1) - s(i, j+1)
        The right hand side are measurements from the source image. The left
        hand side relates different (mostly) unknown pixels in the output
        image. At a high level, for these rows in our system of equations we
        are saying "For this pixel, I don't know its value, but I know that
        its value relative to its neighbors should be the same as it was in
        the source image".

Functions/packages you may find useful:

    scipy.sparse:

        This is the python package we intend for you to use on
        this assignment for sparse matrices. There are several sparse matrix
        storage options that the package offers. It's important to know the
        differences between them:

        https://scipy-lectures.org/advanced/scipy_sparse/storage_schemes.html

        The key takeaway here is that some sparse storage schemes are fast for
        indexing and incremental construction (e.g., LIL, DOK), while others
        are much faster for performing arithmetic operations upon (e.g., CSC,
        CSR). When setting up the matrices for the sparse linear system, it
        may be advantageous to use one storage scheme, then convert to another
        when actually solving the system. Otherwise, running your code may
        take a *very* long time.

    scipy.sparse.linalg:

        It's important to use this module when solving the linear system
        rather than the numpy linalg module, since this one is specifically
        built for handling sparse matrices.

    tqdm.tqdm:

        This is a simple-to-use function that allows for printing
        progress bars. For example, the for loop declared as:

            for i in range(20):

        can be wrapped with tqdm as such:

            for i in tqdm(range(20)):

        with the only difference being that a progress bar is printed.
        This is useful for monitoring the progress of operations that can
        potentially take some time, like setting up the sparse matrices
        of your linear system.
"""

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
import cv2

IS_TRANSPARENT = True
def alpha_blending(source, mask, target):
    """
    Performs alpha blending. 
    Source, mask, and target are all numpy arrays of the same shape 
    (this is ensured by the fix_images function called in main.py).

    Args:
        source - np.array of source image
        mask   - np.array of binary mask. Could also be matte.
        target - np.array of target image

    Returns:
        np.array of blended image
    """

    return (source * mask) + (target * (1 - mask))

def build_laplacian_pyramid_recursive(img, depth, detail_list):
    MAX_DEPTH = 4
    if depth>=MAX_DEPTH: # reached hardcoded low-res limit or max depth
        detail_list.append(img)
        return detail_list
    # blur image
    blur= cv2.GaussianBlur(img,(3,3),0)
    detail = cv2.subtract(img, blur)
    detail_list.append(detail)
    image = cv2.resize(blur, (0, 0), fx=0.5, fy=0.5)
    return build_laplacian_pyramid_recursive(image, depth+1, detail_list)

def build_gaussian_pyramid_recursive(img, depth, blur_list):
    MAX_DEPTH = 4
    #  or b.shape[0]*b.shape[1] < MIN_SIZE_PYRAMID_IMG
    if depth>=MAX_DEPTH: # reached hardcoded low-res limit or max depth
        blur_list.append(img)
        return blur_list
    # blur image
    blur= cv2.GaussianBlur(img,(5,5),0)
    image = cv2.resize(blur, (0, 0), fx=0.5, fy=0.5)
    blur_list.append(blur)
    return build_gaussian_pyramid_recursive(image, depth+1, blur_list)

def reconstruct_laplacian_pyramid(Ls):
    image = Ls[-1]
    for i in range(Ls.__len__() - 2, -1, -1):
        image = cv2.resize(image, (Ls[i].shape[1], Ls[i].shape[0]))
        image = image + Ls[i]
    return image

def laplacian_pyramid_blend(source, mask, target):
    """
    Performs Laplacian pyramid blending (from lab 'compositing'). 
    Source, mask, and target are all numpy arrays of the same shape 
    (this is ensured by the fix_images function called in main.py).

    Args:
        source - np.array of source image
        mask   - np.array of binary mask
        target - np.array of target image

    Returns:
        np.array of blended image
    """
    l1 = build_laplacian_pyramid_recursive(source, 1, [])
    l2 = build_laplacian_pyramid_recursive(target, 1, [])
    gm = build_gaussian_pyramid_recursive(mask, 1, [])
    l_out = []
    for i in range(4):
        l_out.append((gm[i] * l1[i]) + ((1 - gm[i]) * l2[i]))
    output = reconstruct_laplacian_pyramid(l_out)
    return output


def bounds_check_b(b, oneDimIndex, numRows, numCols, col, row, source):
    source_grad = 4*source[row, col]
    if  row+1 < numRows:
        source_grad -= source[row+1, col]

    if  row-1 >= 0:
        source_grad -= source[row-1, col]

    if  col+1< numCols:
        source_grad -= source[row, col+1]
    
    if  col-1 >= 0:
        source_grad -= source[row, col-1]
    return source_grad



def bounds_check_A(A, oneDimIndex, numRows, numCols, col, row, source):
    if row+1 < numRows:
        A[oneDimIndex, (row+1)*source.shape[1] + col] = -1

    if  row-1 >= 0:
        A[oneDimIndex, (row-1)*source.shape[1] + col] = -1

    if  col+1 < numCols:
        A[oneDimIndex, row*source.shape[1] + col+1] = -1
    
    if  col-1 >= 0:
        A[oneDimIndex, row*source.shape[1] + col-1] = -1
    A[oneDimIndex, oneDimIndex] = 4

def poisson_blend_channel(source, mask, target, A, makeA, isAlpha):
    numRows, numCols = source.shape[0], source.shape[1]
    if makeA:
        A = identity(source.shape[0] * source.shape[1]).tolil()
    b = np.zeros(source.shape[0] * source.shape[1])
    for col in range(source.shape[1]):
        for row in range(source.shape[0]):
            # oneDimIndex = col*source.shape[0] + row # acc to unfolding from lecture slides
            oneDimIndex = row*source.shape[1] + col # acc to unfolding from lecture slides
            if mask[row, col] == 0:
                # not under the mask
                b[oneDimIndex] = target[row,col]
            else:
                # under the mask - QUESTION SHOULD I PAD THE SOURCE? - note: all masks have zeroes at the edges so this should be fine
                if makeA:
                    bounds_check_A(A, oneDimIndex, numRows, numCols, col, row, source)
                source_grad = bounds_check_b(b, oneDimIndex, numRows, numCols, col, row, source)
                if isAlpha:
                    target_grad = bounds_check_b(b, oneDimIndex, numRows, numCols, col, row, target)
                    b[oneDimIndex] = max(source_grad, target_grad)
                else:
                    b[oneDimIndex] = source_grad

    A = A.tocsr()
    x = spsolve(A, b)
    # x = x.reshape((source.shape[1], source.shape[0]))
    x = x.reshape(source.shape)
    print("DONE")
    return x, A


# mixing with alpha
def poisson_blend(source, mask, target):
    """
    Performs Poisson blending. Source, mask, and target are all numpy arrays
    of the same shape (this is ensured by the fix_images function called in
    main.py).

    Args:
        source - np.array of source image
        mask   - np.array of binary mask
        target - np.array of target image

    Returns:
        np.array of blended image
    """

    # TODO: Implement this function!
    '''
    We will set up a system of equations A * x = b, where A has as many rows
and columns as there are pixels in our images. Thus, a 300x200 image will
lead to A being 60000 x 60000. 'x' is our output image (a single color
channel of it) stretched out as a vector. 'b' contains two types of known
values:

    (1) For rows of A which correspond to pixels that are not under the
        mask, b will simply contain the already known value from 'target'
        and the row of A will be a row of an identity matrix. Basically,
        this is our system of equations saying "do nothing for the pixels we
        already know".
    (2) For rows of A which correspond to pixels under the mask, we will
        specify that the gradient (actually the discrete Laplacian) in the
        output should equal the gradient in 'source', according to the final
        equation in the webpage:
           4*x(i,j) - x(i-1, j) - x(i+1, j) - x(i, j-1) - x(i, j+1) =
           4*s(i,j) - s(i-1, j) - s(i+1, j) - s(i, j-1) - s(i, j+1)
        The right hand side are measurements from the source image. The left
        hand side relates different (mostly) unknown pixels in the output
        image. At a high level, for these rows in our system of equations we
        are saying "For this pixel, I don't know its value, but I know that
        its value relative to its neighbors should be the same as it was in
        the source image".
    '''
    # x = np.zeros((source.shape[1], source.shape[0],3))
    x = np.zeros(source.shape)
    x[:,:,0], A = poisson_blend_channel(source[:,:,0], mask[:,:,0], target[:,:,0], None, True, False)
    x[:,:,1], _ = poisson_blend_channel(source[:,:,1], mask[:,:,1], target[:,:,1], A, False, False)
    x[:,:,2], _ = poisson_blend_channel(source[:,:,2], mask[:,:,2], target[:,:,2], A, False, False)
    return x


def color_2_Gray(hue, mask, value):
    # convert image to hsv
    # do poinsson blend but target and source = 0 and 2 channel of hsv
    x = np.zeros(hue.shape)
    x, _ = poisson_blend_channel(hue, mask, value, None, True, False)
    return x
    

def poisson_blend_transparent(source, mask, target):
    x = np.zeros(source.shape)
    print(mask.shape)
    x[:,:,2], A = poisson_blend_channel(source[:,:,0], mask, target[:,:,0], None, True, False)
    x[:,:,1], _ = poisson_blend_channel(source[:,:,1], mask, target[:,:,1], A, False, False)
    x[:,:,0], _ = poisson_blend_channel(source[:,:,2], mask, target[:,:,2], A, False, False)
    x[:,:,3], _ = poisson_blend_channel(source[:,:,3], mask, target[:,:,3], A, False, True)
    return x

