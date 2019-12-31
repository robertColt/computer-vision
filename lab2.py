'''
1.       Write an application that loads two images:
a.       Scene image
b.      Logo image

And superposes the logo image over the scene and allows to see through the zones in the logo that do not contain details/information. Hint: use the opencv_logo.png as logo
2.       Implement the fill contour using morphological operations algorithm presented during lecture 5.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def write_on_img(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape
    cv2.putText(img, text, (10, h - 20), font, 1, (255, 255, 250), 2, cv2.LINE_AA)


def flood_fill_Cv(img):
    flood_fill_img = img.copy()
    h, w = flood_fill_img.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_fill_img, mask, (0, 0), 255)
    flood_fill_inv = cv2.bitwise_not(flood_fill_img)
    flood_filled = img | flood_fill_inv
    return flood_filled


def flood_fill_morpho():
    # create cross shape 3x3
    b = np.zeros((3, 3), dtype='uint8')
    b[1] = np.ones((1, 3), dtype='uint8')
    b[:, 1] = np.ones((3,), dtype='uint8')

    # draw a contour to fill
    h, w = 7, 7
    x_start, y_start = 1, 1
    rect_height = 4

    # zero background
    x = np.zeros((h, w), dtype='uint8')
    #black recrangle on top
    cv2.rectangle(x, (x_start, y_start), (x_start + rect_height, x_start + rect_height), color=(255, 255, 255),
                  thickness=1)
    # normalize to binary
    x = x // 255

    x_inv = cv2.bitwise_not(x).astype('uint8') // 255
    xk_list = [x, x_inv]
    xk_prev = np.zeros(x.shape, dtype='uint8')

    # make initial point inside the contour 1
    xk_prev[y_start + 1, x_start + 1] = 1
    i = 0
    while True:
        xk_list.append(xk_prev)
        xk = cv2.morphologyEx(xk_prev, cv2.MORPH_DILATE, kernel=b).astype('uint8')
        xk = cv2.bitwise_and(xk, x_inv).astype('uint8')
        if np.array_equal(xk, xk_prev):
            print('break', i)
            break
        i = i + 1
        xk_prev = xk

    xk_list.append(cv2.bitwise_or(x, xk))
    progress = np.hstack(xk_list)
    plt.imshow(progress, cmap='binary')
    ax = plt.gca()
    ax.grid(which='minor', color='b', linestyle='-', linewidth=1)
    plt.show()


img = cv2.imread('opencv.png', cv2.IMREAD_GRAYSCALE)
thresh, img_thresh = cv2.threshold(img, 10, 255, type=cv2.THRESH_BINARY)
dilation = cv2.morphologyEx(img_thresh, cv2.MORPH_DILATE, kernel=np.ones((5, 5)))
erosion = cv2.morphologyEx(img_thresh, cv2.MORPH_ERODE, kernel=np.ones((5, 5)))
gradient = dilation - erosion
flood_filled = flood_fill_Cv(gradient)
flood_fill_morpho()

write_on_img(img_thresh, 'original')
write_on_img(dilation, 'dilation')
write_on_img(erosion, 'erosion')
write_on_img(gradient, 'gradient')
write_on_img(flood_filled, 'flood filled')
# write_img(flood_filled_morpho, 'flood filled morpho')
final_img = np.hstack([img_thresh, dilation, erosion, gradient, flood_filled])
plt.imshow(final_img, cmap='binary')
plt.show()
