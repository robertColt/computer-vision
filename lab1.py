import cv2
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt


def overlay_images(background_img, transparent_img):
    '''
    ooverlays the transparent img ove rthe backgroung image
    :param background_img:
    :param transparent_img:
    :return:
    '''
    # mask for important parts in image
    img_mask = transparent_img[:, :, -1] // 255
    img = transparent_img[:, :, :-1]

    # mask for background parts hidden by the img mask (inverse of img_mask)
    background_mask = np.bitwise_not(img_mask) // 255

    background_snipped = cv2.bitwise_and(background_img, background_img, mask=background_mask)
    img_snipped = cv2.bitwise_and(img, img, mask=img_mask)
    overlapped = cv2.add(background_snipped, img_snipped)

    return overlapped


video_capture = cv2.VideoCapture(0)
opencv_logo = cv2.imread('opencv.png', cv2.IMREAD_UNCHANGED)
logo_h, logo_w, logo_c = opencv_logo.shape

while (True):
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    frame[:logo_h, :logo_w] = overlay_images(frame[:logo_h, :logo_w], opencv_logo)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = np.stack((gray_img,) * 3, axis=-1)
    blurred = cv2.blur(frame, (5, 5))

    edges = cv2.Canny(frame, 100, 200)
    edges = np.stack((edges,) * 3, axis=-1)
    images = np.hstack([np.vstack([frame, gray_img]),
                        np.vstack([blurred, edges])])

    cv2.imshow('Webcam - Press q to quit', images)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
