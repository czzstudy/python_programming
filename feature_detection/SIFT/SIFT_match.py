# this program is to use SIFT to detect the feature and match for registration
import numpy as np
import cv2 

Img = cv2.imread("F://study//python//python_programming//resource//test.jpg", flags=2)  # read the image in gray level
