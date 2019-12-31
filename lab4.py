'''
Write an application that uses a dataset of images (of at least) 30-50 images downloaded from internet with various content (landscapes, objects, people, etc). The application should take a new image as input argument and determine the set of similar images from the database:

1.       using all histogram comparison metrics available in OpenCV.
2.       By using a color reduction mechanism to transform similar colors from an equivalence class to its representative color and conducting the histogram comparison on the reduced color spaces

Compare the results and explain.
'''

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def read_img_and_hist(img_path, new_size, reduce_2gray):
    img = cv2.imread(img_path)
    target_color_space = cv2.COLOR_BGR2RGB if not reduce_2gray else cv2.COLOR_BGR2GRAY
    img = cv2.cvtColor(img, target_color_space)
    img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_CUBIC)

    n_channels = 1 if target_color_space == cv2.COLOR_RGB2GRAY else 3
    channels = [i for i in range(n_channels)]
    bins_per_channel = [8] * n_channels
    bin_ranges = [0, 256] * n_channels
    img_hist = cv2.calcHist(img, channels, None, bins_per_channel, bin_ranges)
    img_hist = cv2.normalize(img_hist, img_hist)
    return img, img_hist


def show_images(images, plot_title, n_images, cols=1):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    fig = plt.figure()
    plt.title(plot_title)
    for n, (compare_score, image) in enumerate(images):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        title = '{:.2f}'.format(compare_score)
        if n == 0: title = 'original ' + title
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


class ImageHistComparator:
    def __init__(self, default_size, reduce_colors):
        self.default_size = default_size
        self.reduce_colors = reduce_colors
        self.img_database = []
        self.img_hists = []
        self.comparison_methods = [cv2.HISTCMP_CORREL,
                                   cv2.HISTCMP_BHATTACHARYYA,
                                   cv2.HISTCMP_CHISQR,
                                   cv2.HISTCMP_INTERSECT
                                   ]
        self.comparison_methods_similar_lambda = [lambda list_: reversed(sorted(list_, key=lambda x: x[0])),
                                                  lambda list_: sorted(list_, key=lambda x: x[0]),
                                                  lambda list_: sorted(list_, key=lambda x: x[0]),
                                                  lambda list_: reversed(sorted(list_, key=lambda x: x[0]))
                                                  ]
        self.comparison_methods_str = 'cv2.HISTCMP_CORREL, \
                                   cv2.HISTCMP_BHATTACHARYYA, \
                                   cv2.HISTCMP_CHISQR, \
                                   cv2.HISTCMP_INTERSECT'.split(',')

    def init_from_dir(self, dir_path):
        img_paths = ['{}{}'.format(dir_path, img_path) for img_path in os.listdir(dir_path)]
        for img_path in img_paths:
            img, img_hist = read_img_and_hist(img_path, self.default_size, self.reduce_colors)
            self.img_database.append(img)
            self.img_hists.append(img_hist)
        print(self.img_hists[0].shape)
        print('Initialized with {} images'.format(len(img_paths)))

    def show_similar(self, img_path):
        img, img_hist = read_img_and_hist(img_path, self.default_size, self.reduce_colors)
        for comp_method, comp_method_str, similar_lambda in zip(self.comparison_methods, self.comparison_methods_str,
                                                                self.comparison_methods_similar_lambda):
            compare_score_self = cv2.compareHist(img_hist, img_hist, comp_method)
            compare_scores = []
            compare_images = []
            for i, img_hist_database in enumerate(self.img_hists):
                compare_score = cv2.compareHist(img_hist, img_hist_database, comp_method)
                compare_scores.append(compare_score)
                compare_images.append(self.img_database[i])

            similar_images_and_scores = similar_lambda(zip(compare_scores, compare_images))
            similar_images_and_scores = [(compare_score_self, img)] + [x for x in similar_images_and_scores]
            show_images(similar_images_and_scores, comp_method_str, n_images=len(self.img_database) + 1, cols=3)


if __name__ == '__main__':
    dir_name = 'img_dataset/'
    dir_name_test = 'test_img/'
    img_size = 500
    image_hist_comparator = ImageHistComparator(img_size, reduce_colors=False)
    image_hist_comparator.init_from_dir(dir_name)
    test_images_path = [dir_name_test + img_path for img_path in os.listdir(dir_name_test)]
    img_id = 0

    image_hist_comparator.show_similar(test_images_path[img_id])

    # counter = 0
    # img_paths = [dir_name + img_name for img_name in os.listdir(dir_name)]
    # for img_path in img_paths:
    #     file_ext = img_path.split('.')[-1]
    #     new_file_name = dir_name + str(counter) + '.' + file_ext
    #     os.rename(img_path, new_file_name)
    #     counter += 1
