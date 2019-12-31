'''
Write an application that uses morphological operators to extract corners from an image using the following algorithm:

1.       R1 = Dilate(Img,cross)
2.       R1 = Erode(R1,Diamond)
3.       R2 = Dilate(Img,Xshape)
4.       R2 = Erode(R2,square)
5.       R = absdiff(R2,R1)
6.       Display(R)

Transform the input image to make it compatible with binary operators and display the results imposed over the original image. Apply the algorithm at least on the following images Rectangle and Building.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

size = 5
struct_shape = (size, size)
# CROSS shaped kernel
cross = np.zeros(struct_shape, dtype='uint8')
one_array = np.ones((size,), dtype='uint8')
cross[:, size//2] = one_array
cross[size//2] = one_array

# SQUARE shaped kernel
square = np.ones(struct_shape, dtype='uint8')

# X shaped kernel
x_shape = np.zeros(struct_shape, dtype='uint8')
np.fill_diagonal(x_shape, 1)
np.fill_diagonal(np.fliplr(x_shape), 1)

# diamond shaoe
diamond = np.zeros((5,5), dtype='uint8')
diamond[2] = np.ones((5, ), dtype='uint8')
diamond[:, 2] = np.ones((5,), dtype='uint8')
diamond[1:4, 1:4] = np.ones((3,3), dtype='uint8')

kernels = np.hstack([cross, square, x_shape])

# plt.imshow(diamond, cmap='binary')
# plt.show()


def corner_draw(img, src_img):
    r1 = cv2.dilate(img, cross)
    r1 = cv2.erode(r1, diamond)
    r2 = cv2.dilate(img, x_shape)
    r2 = cv2.erode(r2, square)
    corner_map = cv2.absdiff(r2, r1)
    src_img_copy = src_img.copy()
    for y, row in enumerate(corner_map):
        for x, col in enumerate(row):
            if col == 1:
                cv2.circle(src_img_copy, (x, y), radius=6, color=(255, 255, 255), thickness=1)
    return corner_map, src_img_copy


def img_gradient(img_thresh):
    morpho_ksize = 5
    morpho_ksize = (morpho_ksize, morpho_ksize)
    dilation = cv2.dilate(img_thresh, np.ones(morpho_ksize))
    erosion = cv2.erode(img_thresh, np.ones(morpho_ksize))
    return dilation - erosion


def process_img(img_path):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)
    print(img.shape)

    thresh = 140
    img_thresh_adaptive = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 33, 20)
    ret, img_thresh = cv2.threshold(img, thresh, 1, cv2.THRESH_BINARY_INV)

    img_gradient_adaptive = img_gradient(img_thresh_adaptive)
    img_gradient_normal = img_gradient(img_thresh)

    img_gradient_adaptive_corner_map, corner_img_gradient_adaptive = corner_draw(img_gradient_adaptive, src_img=img)
    img_gradient_corner_map, corner_img_gradient = corner_draw(img_gradient_normal, src_img=img)

    images = [img_thresh_adaptive, img_gradient_adaptive, img_gradient_adaptive_corner_map, corner_img_gradient_adaptive,\
              img_thresh, img_gradient_normal, img_gradient_corner_map, corner_img_gradient]
    titles = ['adaptive thresh', 'gradient', 'corner map', 'corners',\
              'normal thresh', 'gradient', 'corner map', 'corners']
    cmaps = ['binary', 'binary', 'binary', 'gray']
    cmaps = cmaps*2
    for i in range(len(images)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i], cmap=cmaps[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


process_img('corner_image1.png')
process_img('corner_image2.png')

