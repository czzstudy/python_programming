# this program is to use mutual information to register
# refer: https://blog.csdn.net/hujingshuang/article/details/47910949

import cv2
import numpy as np
import math
import time

# entropy calculation for one image
def Entropy_calc(img_matrix):
    tmp = np.zeros(256)
    Entropy = 0
    num = 0
    for i in range(img_matrix.shape[0]):
        for j in range(img_matrix.shape[1]):
            value = img_matrix[i, j]
            tmp[value] = tmp[value] + 1  
            num = num + 1
    for i in range(tmp.size):
        p = tmp[i]/num
        if p == 0:
            Entropy = Entropy
        else:
            Entropy = Entropy - p*(math.log(p))
    return Entropy

# joint entropy calculation for two images
def Joint_Entropy_calc(img1_matrix, img2_matrix):
    if img1_matrix.shape != img2_matrix.shape:
        print("img1's size isn't equal to img2's!")
        return 0
    else:
        tmp = np.zeros((256, 256), dtype=float)
        Joint_Entropy = 0
        num = 0
        for i in range(img1_matrix.shape[0]):
            for j in range(img1_matrix.shape[1]):
                value1 = img1_matrix[i, j]
                value2 = img2_matrix[i, j]
                tmp[value1, value2] = tmp[value1, value2] + 1  
                num = num + 1
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                pij = tmp[i, j]/num
                if pij == 0:
                    Joint_Entropy = Joint_Entropy
                else:
                    Joint_Entropy = Joint_Entropy - pij*math.log(pij)
        return Joint_Entropy

# Mutual Information Registration Algorithm
def MI_reg(Img, Img_crop):
    Img_matrix = np.matrix(Img)
    Img_crop_matrix = np.matrix(Img_crop)
    Img_height, Img_width = Img_matrix.shape
    Img_crop_height, Img_crop_width = Img_crop_matrix.shape
    MI_max = 0
    MI_max_i = 0
    MI_max_j = 0
    for i in range(Img_height-Img_crop_height+1):
        for j in range(Img_width-Img_crop_width+1):
            Img_win = Img_matrix[i:Img_crop_height+i, j:Img_crop_width+j]
            MI = Entropy_calc(Img_win) + Entropy_calc(Img_crop) - Joint_Entropy_calc(Img_win, Img_crop)
            if MI > MI_max:
                MI_max = MI
                MI_max_i = i
                MI_max_j = j
    return [MI_max_i, MI_max_j, MI_max]

# Normalization Mutual Information Registration Algorithm
def NMI_reg(Img, Img_crop):
    Img_matrix = np.matrix(Img)
    Img_crop_matrix = np.matrix(Img_crop)
    Img_height, Img_width = Img_matrix.shape
    Img_crop_height, Img_crop_width = Img_crop_matrix.shape
    NMI_max = 0
    NMI_max_i = 0
    NMI_max_j = 0
    for i in range(Img_height-Img_crop_height+1):
        for j in range(Img_width-Img_crop_width+1):
            Img_win = Img_matrix[i:Img_crop_height+i, j:Img_crop_width+j]
            NMI = (Entropy_calc(Img_win) + Entropy_calc(Img_crop))/Joint_Entropy_calc(Img_win, Img_crop)
            if NMI > NMI_max:
                NMI_max = NMI
                NMI_max_i = i
                NMI_max_j = j
    return [NMI_max_i, NMI_max_j, NMI_max]

# Entropy Corrleation Coefficient Registration Algorithm
def ECC_reg(Img, Img_crop):
    Img_matrix = np.matrix(Img)
    Img_crop_matrix = np.matrix(Img_crop)
    Img_height, Img_width = Img_matrix.shape
    Img_crop_height, Img_crop_width = Img_crop_matrix.shape
    ECC_max = 0
    ECC_max_i = 0
    ECC_max_j = 0
    for i in range(Img_height-Img_crop_height+1):
        for j in range(Img_width-Img_crop_width+1):
            Img_win = Img_matrix[i:Img_crop_height+i, j:Img_crop_width+j]
            MI = Entropy_calc(Img_win) + Entropy_calc(Img_crop) - Joint_Entropy_calc(Img_win, Img_crop)
            ECC = 2*MI/(Entropy_calc(Img_win) + Entropy_calc(Img_crop))
            if ECC > ECC_max:
                ECC_max = MI
                ECC_max_i = i
                ECC_max_j = j
    return [ECC_max_i, ECC_max_j, ECC_max]

if __name__=='__main__':
    Img = cv2.imread("F://study//python//python_programming//resource//test.jpg", flags=2)  # read the image in gray level
    Img_crop = cv2.imread('F://study//python//python_programming//resource//test_crop.jpg', flags=2)  # read the crop image as template
    starttime = time.time()
    print(MI_reg(Img, Img_crop))  # use 2.6hours to run 
    #print(NMI_reg(Img, Img_crop))
    #print(ECC_reg(Img, Img_crop))
    endtime = time.time()
    print(round(endtime-starttime), 'secs', ' ', round(endtime-starttime)/3600, 'hours')