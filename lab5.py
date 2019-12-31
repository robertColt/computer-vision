'''
Write an algorithm for line and circle detection using the Hough method and apply it to images that contain potential lines and circles. Vary the arguments given to the algorithms and see the results.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_lines(img, lines_list):
    for line in lines_list:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def plot(ax, img, title, gray=False):
    if gray:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    ax.set_title(title)


def detect_lines(image_paths):
    for img_path in image_paths:
        img = cv2.imread(img_path, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ksize = 3
        img_gray = cv2.GaussianBlur(img_gray, (ksize, ksize), 1)
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3, L2gradient=True)
        fig, ax = plt.subplots(1, 2)
        plt.tight_layout()
        plot(ax[0], img, 'original')

        lines = cv2.HoughLines(edges, 2, np.pi / 180, 200)
        new_img = img.copy()
        draw_lines(new_img, lines)
        plot(ax[1], new_img, 'Lines', gray=True)
        plt.show()


def detect_circles(images_paths):
    for img_path in images_paths:
        img = cv2.imread(img_path, 0)
        img = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,
                                   minDist=70, #min dist between radiuses of the circles
                                   param1=50,
                                   param2=40, # the smaller the more false circles it may detect
                                   minRadius=20, maxRadius=55) # 80 for overall results

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        plt.imshow(cimg)
        plt.show()


line_images = ['line_img3.jpg', 'line_img1.jpg', 'line_img2.jpg']
circle_images = ['circle_image1.jpeg', 'circle_image2.jpg', 'circle_image3.jpeg']

detect_circles(circle_images)
detect_lines(line_images)
