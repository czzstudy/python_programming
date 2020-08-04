# use this program to do some simple processing to the img and save it for the following analysis

import cv2
import numpy as np


Img = cv2.imread('F://study//python//python_programming//resource//test.jpg', flags=2)  # read the image in gray level
Img_array = np.matrix(Img)
Img_height, Img_width = Img_array.shape
print(Img_height, Img_width, Img_array.shape[0], Img_array.shape[1])
# crop the img and save it
Img_crop = Img[150:300, 150:300]
cv2.imshow('Img_crop', Img_crop)
cv2.waitKey(3000)
cv2.imwrite('F://study//python//python_programming//resource//test_crop.jpg', Img_crop)
# rotation to the img and save it
M_rotation = cv2.getRotationMatrix2D((Img_width/2, Img_height/2), 90, 1)
Img_rotated = cv2.warpAffine(Img, M_rotation, (Img_width, Img_height))
cv2.imshow('Img_rotated', Img_rotated)
cv2.waitKey(3000)
cv2.imwrite('F://study//python//python_programming//resource//test_rotated.jpg', Img_rotated)