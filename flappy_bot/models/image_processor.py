from typing import Tuple
import attr
import matplotlib.pyplot as plt
import numpy
import cv2
from functools import reduce
import time
import math
from structlog import get_logger

logger = get_logger(__name__)


@attr.s
class ImageProcessor:

    @staticmethod
    def find_match_with_template(image: numpy.array, template: numpy.array, threshold: float=0.8) -> numpy.array:
        start_time = time.time()
        # Template matching is very good for exact matches.
        match = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        match = numpy.where(match >= threshold)
        logger.debug("[find_match_with_template]", runtime=time.time()-start_time)
        return match

    @staticmethod
    def find_by_color(image: numpy.array, upper_bound: numpy.array, lower_bound: numpy.array) -> numpy.array:
        start_time = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(image, upperb=upper_bound, lowerb=lower_bound)
        points = cv2.findNonZero(mask)
        logger.debug("[find_by_color]", runtime=time.time()-start_time)
        #cv2.imshow('frame', image)
        #cv2.imshow('mask', mask)
        #plt.cla()
        #plt.imshow(mask)
        #plt.imshow(image)
        #plt.pause(0.00001)
        #if points.size > 0:
        #    cv2.imshow('res', points)
        return points, mask

    @staticmethod
    def find_match_with_features(containing_image: numpy.array, target_image: numpy.array):
        # https://www.docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

        # Create orb detector??
        orb = cv2.ORB_create(edgeThreshold=4, nfeatures=5000)

        # Look for keypoints and descriptors
        keypoints_target_image, descriptors_target_image = orb.detectAndCompute(target_image, None)
        keypoints_containing_image, descriptors_containing_image = orb.detectAndCompute(containing_image, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(descriptors_target_image, descriptors_containing_image)
        # Apply ratio test
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)[:2]



        # make a list of points
        #groups = list(map(lambda y:[x for x in matches if x.distance - y.distance < max_distance], matches))
        #group = matches[:2]

        # largest list first
        average_location = numpy.average([(keypoints_containing_image[x.trainIdx].pt[0], keypoints_containing_image[x.trainIdx].pt[1]) for x in matches], axis=0)
        average_location = average_location.astype(int)
        #print(average_location)

        # hacky, get the average location of the object:
        #x = [keypoints_containing_image[x.queryIdx].pt[1] for x in groups[0]]
        #x = reduce(lambda a, b: a + b, x) / len(x)

        #y = [keypoints_containing_image[x.queryIdx].pt[0] for x in groups[0]]
        #y = reduce(lambda a, b: a + b, y) / len(y)

        img3 = cv2.drawMatches(target_image, keypoints_target_image, containing_image, keypoints_containing_image, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        #return int(x), int(y)
        return img3, average_location[0], average_location[1]

    @staticmethod
    def _show_orb_keypoints(image: numpy.array):
        orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, edgeThreshold=15)
        kp = orb.detect(image)
        img2_kp = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
        plt.figure()
        plt.imshow(img2_kp)
        plt.show()


    @staticmethod
    def draw_match_on_image(image: numpy.array, template: numpy.array, match: numpy.array):
        w, h = template.shape[::-1]

        for pt in zip(*match[::-1]):
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        return image

    @staticmethod
    def draw_line_on_image(image: numpy.array, start: Tuple[int, int], end: Tuple[int, int]):
        cv2.line(image, start, end, (0, 0, 255), 2)
        return image

    @staticmethod
    def draw_circle(image: numpy.array, center: Tuple[int, int], radius: int):
        cv2.circle(image, center, radius, (150, 255, 255), -1)
        return image

