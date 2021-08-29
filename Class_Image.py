# -*- coding: utf-8 -*-
"""
Definition file of the Image an Panorama classes.

The Image class is used to represents a digital image and to compute some properties such as its blurriness.
The Panorama class inherits the Image class and add specific methods for this type of image.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

import cv2
import trueskill
from GPSPhoto import gpsphoto
from skimage import color
from keras.preprocessing import image as ki


from PIL import Image as PIL_Image
import requests
# ----------------------------------------------------------------------------------------------------------------------
# Class definition
# ----------------------------------------------------------------------------------------------------------------------
class Image:
    """
    Definition of the Image class.

    The Image class is used to represents a digital image and to compute some properties such as its blurriness.
    """
    def __init__(self, path, build_array=False, build_green_segmentation=False, green_seg_model=None,
                 build_snow_segmentation=False, snow_seg_model=None):
        """

        :param path:
        :type path:
        :param build_array:
        :type build_array:
        """
        self.path = path
        self.basename = os.path.basename(self.path)

        self.green_segmentation = np.array([[]])
        self.green_ratio = "Not computed"
        self.snow_segmentation = np.array([[]])
        self.trueskill_rating = trueskill.Rating()
        self.ranking_score = 999.

        self.overexposed = "Not computed"
        self.wrong_exposed = "Not computed"
        self.under_ratio = "Not computed"
        self.blurry = "Not computed"

        self.snow_segmentation = np.array([[]])
        self.snowy = "Not computed"

        self.lat = 0
        self.lon = 0
        self.key = ""
        self.heading = 0
        self.global_key = self.key + "_" + str(self.heading)
        self.datetime = ""

        if build_array:
            self.array = self.imread(self.path)
            self.shape = self.array.shape
            self.width = self.shape[0]
            self.height = self.shape[1]
        else:
            self.array = np.array([[]])
            self.shape = (0, 0)
            self.width = self.shape[0]
            self.height = self.shape[1]

        if build_green_segmentation:
            self.green_seg_img_size = green_seg_model.input_shape[1]
            p_img = self.preprocess_image(self.green_seg_img_size)
            self.green_segmentation = green_seg_model.predict(np.expand_dims(p_img, axis=0))[0]

        if build_snow_segmentation:
            self.snow_seg_img_size = snow_seg_model.input_shape[1]
            p_img = self.preprocess_image(self.snow_seg_img_size)
            self.snow_segmentation = snow_seg_model.predict(np.expand_dims(p_img, axis=0))[0]

    def __str__(self):
        s = "=== Image information === "
        s += "\nName : " + os.path.basename(self.path)
        s += "\nDirectory : " + os.path.dirname(self.path)
        s += "\nPath : " + self.path
        s += "\nOverexposed : " + str(self.overexposed)
        s += "\nBad exposure  : " + str(self.wrong_exposed)
        s += "\nGreen ratio : " + str(self.green_ratio)
        s += "\nUnder green ratio : " + str(self.under_ratio)
        s += "\nBlurry : " + str(self.blurry)
        s += "\nShape : " + str(self.shape)
        return s
    
    #Give image array from url or filepath
    def imread(self, path):
        if path.startswith("http"):
            i = PIL_Image.open(requests.get(path, stream=True).raw)
            return np.array(i)
        return cv2.imread(path)
    
    # Image analyses methods
    def is_overexposed(self):
        """
        Checks if the image is overexposed.

        It uses the overexposure definition from the following article:
            Correcting Over-Exposure in Photographs
            Dong Guo, Yuan Cheng, Shaojie Zhuo and Terence Sim
            School of Computing, National University of Singapore, 117417
            DOI : 10.1109/CVPR.2010.5540170
        :return: True if the image is overexposed, False if it is not
        :rtype: bool
        """
        # Variable initialization
        msky = 0
        cmp_sky = 0

        # Resize image to predictions size
        rgb = cv2.resize(self.array, (self.green_seg_img_size, self.green_seg_img_size))
        lab = color.rgb2lab(rgb)

        # Compute mSky
        for i in range(self.green_seg_img_size):
            for j in range(self.green_seg_img_size):
                index_max = np.argmax(self.green_segmentation[i, j])
                if index_max == 3:
                    msky += compute_m(lab[i, j])
                    cmp_sky += 1
        if cmp_sky == 0:
            self.overexposed = False
            return False
        else:
            msky /= cmp_sky
            if msky > 0.85 and cmp_sky > 0.3 * self.green_seg_img_size * self.green_seg_img_size:
                self.overexposed = True
                return True
            else:
                self.overexposed = False
                return False

    def is_wrong_exposed(self):
        """
        Check if an image has an exposure problem caused by a bimodal histogram

        :return: True if the image is wrong exposed, False if it is not
        :rtype: bool
        """
        # Variables initialization
        bad = False

        # Compute image array and change to HSV colorspace
        bgr = self.imread(self.path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Compute histogram
        hist_full_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        total = np.sum(hist_full_v)

        ten_mid = np.sum(hist_full_v[50:-50])
        prop_mid = ten_mid / total

        # Check proportions
        if prop_mid < 0.50:
            bad = True
            # print(prop_end + prop_begin)

        self.wrong_exposed = bad
        return bad

    def is_under_green_ratio(self, threshold):
        """
        Check if the image green ratio is under a threshold

        :param threshold: threshold value
        :type threshold: float
        :return: True if the image's green ration is under the threshold, False if it is not
        :rtype: bool
        """
        # Variable initialization
        cmp_green = 0
        cmp_black = 0

        # Compute green ratio
        for i in range(self.green_segmentation.shape[0]):
            for j in range(self.green_segmentation.shape[1]):
                proba = self.green_segmentation[i][j]
                index_max = np.argmax(proba)
                if index_max == 0:
                    cmp_green += 1
                elif index_max == 1:
                    cmp_black += 1

        # Compare ratio value to threshold
        if cmp_black == 0:
            self.under_ratio = False
            self.green_ratio = "No bad elements :)"
            return False
        else:
            self.green_ratio = cmp_green / cmp_black
            if self.green_ratio < threshold:
                self.under_ratio = True
                return True
            else:
                self.under_ratio = False
                return False

    def is_blurry(self, threshold=80.):
        """
        Check if the image is blurry comparing its variance of Laplacian to a threshold.

        :param threshold: threshold value
        :type threshold: float
        :return: True if the image is blurry, False if it is not
        :rtype: bool
        """
        # Compute variance of Laplacian
        v = variance_of_laplacian(self.array)

        # Compare value
        if v < threshold:
            self.blurry = True
            return True
        else:
            self.blurry = False
            return False

    def is_snowy(self):
        """
        Checks is the image contains snow.

        :return: True if the image has snow, False if it has not
        :rtype: bool
        """
        # Check snow probability
        for i in range(self.snow_segmentation.shape[0]):
            for j in range(self.snow_segmentation.shape[1]):
                proba = self.snow_segmentation[i][j]
                index_max = np.argmax(proba)
                if index_max == 1:
                    self.snowy = True
                    return True
        self.snowy = False
        return False

    # Process image methods
    def preprocess_image(self, img_size):
        """
        Create, normalize and resize the image array

        :param img_size: image output size, corresponding to input size of the Keras model.
        :type img_size: int
        :return: preprocessed image, resized and normalized
        :rtype: np.array
        """
        img = self.imread(self.path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img, (img_size, img_size))
        img_norm = normalized(img_resized)
        return img_norm

    # Retrieving information methods
    def get_info_from_filename(self):
        """
        Gets image global properties from its filename.

        :return: images properties
        :rtype: list
        """
        self.lon = self.basename[0:10]
        self.lat = self.basename[11:20]
        self.key = self.basename[21:43]
        self.global_key = self.basename[21:47]
        self.datetime = self.basename[48:67]
        self.heading = self.basename[44:47]
        return self.lon, self.lat, self.key, self.heading, self.datetime, self.global_key

    def get_global_key_from_filename(self):
        """
        Get image key from its filename

        :return: image key
        :rtype: str

        """
        self.global_key = self.basename[21:47]
        return self.global_key

    def get_data_from_exif(self):
        """
        Get image information from its EXIF metadata.

        :return: images information
        :rtype: tuple
        """
        dic = gpsphoto.getGPSData(self.path)
        self.lat = "{0:.6f}".format(dic["Latitude"])
        self.lon = "{0:.6f}".format(dic["Longitude"])
        self.datetime = exif_to_sql(dic['Date'], dic['UTC-Time'])
        return self.lat, self.lon, self.datetime

    def get_image_snow_segmentation(self):
        """
        Return the image snow segmentation color array.

        :return: snow segmentation color array.
        :rtype: np.array
        """
        # Segmentation image initialization
        data = np.zeros((self.snow_segmentation.shape[1], self.snow_segmentation.shape[1], 3))

        # Segmentation build
        for i in range(self.snow_segmentation.shape[1]):
            for j in range(self.snow_segmentation.shape[1]):
                proba = self.snow_segmentation[i][j]
                index_max = np.argmax(proba)

                # Define pixel colors according to the segmentation type
                if index_max == 0:
                    data[i, j] = [0, 0, 0]
                elif index_max == 1:
                    data[i, j] = [255, 255, 255]

        return data

    def get_image_green_segmentation(self):
        """
        Return the image green segmentation color array.

        :return: green segmentation color array.
        :rtype: np.array
        """
        # Segmentation image initialization
        data = np.zeros((self.green_segmentation.shape[1], self.green_segmentation.shape[1], 3))

        # Segmentation build
        for i in range(self.green_segmentation.shape[1]):
            for j in range(self.green_segmentation.shape[1]):
                proba = self.green_segmentation[i][j]
                index_max = np.argmax(proba)

                # Define pixel colors according to the segmentation type
                if index_max == 0:
                    data[i, j] = [0, 128, 0]
                elif index_max == 1:
                    data[i, j] = [0, 0, 0]
                elif index_max == 2:
                    data[i, j] = [200, 200, 200]
                elif index_max == 3:
                    data[i, j] = [0, 0, 255]

        return data

    # Plot image information
    def plot_original_and_segmentation(self, save_folder, seg_type):
        """
        Save in a png file the image and a segmentation.

        :param save_folder: folder where the output file is saved
        :type save_folder:  str
        :param seg_type: segmentation type
        :type seg_type: str
        """
        # Figure initialization
        fig = plt.figure(figsize=(20,10))

        # Add original image
        fig.add_subplot(1, 2, 1)
        b, g, r = cv2.split(self.array)
        rgb_img = cv2.merge([r, g, b])
        plt.imshow(rgb_img, aspect="auto")

        # Add segmentation image
        fig.add_subplot(1, 2, 2)
        if seg_type == "snow":
            plt.imshow(ki.array_to_img(self.get_image_snow_segmentation()), aspect='auto')
        elif seg_type == "green":
            plt.imshow(ki.array_to_img(self.get_image_green_segmentation()), aspect='auto')

        # Save figure
        plt.savefig(os.path.join(save_folder, "{}_segmentation.png".format(seg_type)))

    def plot_histo_hsv(self, save_folder):
        """
        Save in a png file the images and its Value histogram from the HSV colorspace.

        :param save_folder: folder where the output file is saved
        :type save_folder:  str
        """
        # Create figure
        fig_name = "Histogram_{}".format(self.basename)
        fig = plt.figure(fig_name, figsize=(20, 10))

        # Add image to figure
        fig.add_subplot(1, 2, 1)
        b, g, r = cv2.split(self.array)
        rgb_img = cv2.merge([r, g, b])
        plt.imshow(rgb_img, aspect="auto")

        # Add histogram plot
        fig.add_subplot(1, 2, 2)
        hsv = cv2.cvtColor(self.array, cv2.COLOR_BGR2HSV)
        hist_full_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        plt.xlim([0, 256])
        plt.plot(hist_full_v)

        # Save figure
        plt.savefig(os.path.join(save_folder, fig_name), bbox_inches='tight')


def normalized(rgb):
    """
    Normalize the values of an RGB image
    :param rgb: rgb image array
    :type rgb: np.array
    :return: normalized rgb image array
    :rtype: np.array
    """
    # Variable initialization
    norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

    # Get channel order for cv2
    b = rgb[:, :, 0]
    g = rgb[:, :, 1]
    r = rgb[:, :, 2]

    # Normalize values thanks to histogram equalization
    norm[:, :, 0] = cv2.equalizeHist(b)
    norm[:, :, 1] = cv2.equalizeHist(g)
    norm[:, :, 2] = cv2.equalizeHist(r)
    return norm