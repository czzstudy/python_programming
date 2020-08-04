# this program is to test traditional registration algorithms by gray template matching
# CSDN: https://blog.csdn.net/hujingshuang/article/details/47759579, https://blog.csdn.net/hujingshuang/article/details/47803791, https://blog.csdn.net/hujingshuang/article/details/48140397


import cv2
import numpy as np
import random

# Mean Absolute Differences calculation
def MAD_calc(Img_win, Img_tem):
    if Img_win.shape != Img_tem.shape:
        print('子图和模板尺寸不匹配')
        return 0
    else:
        diff = Img_win - Img_tem
        diff_height, diff_width = diff.shape
        sum = np.sum(abs(diff))
        return sum/(diff_height*diff_width)

# Mean Absolute Differences registration algorithm
def MAD_reg(Img, Img_crop):
    Img_matrix = np.matrix(Img)
    Img_crop_matrix = np.matrix(Img_crop)
    Img_height, Img_width = Img_matrix.shape
    Img_crop_height, Img_crop_width = Img_crop_matrix.shape
    MAD_value_min = 10000
    MAD_value_min_i = 0 
    MAD_value_min_j = 0
    for i in range(Img_height-Img_crop_height+1):
        for j in range(Img_width-Img_crop_width+1):
            Img_win = Img_matrix[i:Img_crop_height+i, j:Img_crop_width+j]
            MAD_value = MAD_calc(Img_win, Img_crop_matrix)
            if MAD_value < MAD_value_min:
                MAD_value_min = MAD_value
                MAD_value_min_i = i
                MAD_value_min_j = j
            if MAD_value_min == 0:
                break
    return [MAD_value_min_i, MAD_value_min_j, MAD_value_min]

# Sequential Similiarity Detection Algorithm calculation
def SSDA_calc(Img_win, Img_tem, Threshold):
    if Img_win.shape != Img_tem.shape:
        print('子图和模板尺寸不匹配')
        return 0
    else:
        E_win = np.sum(Img_win)/(Img_win.shape[0]*Img_win.shape[1])
        E_tem = np.sum(Img_tem)/(Img_tem.shape[0]*Img_tem.shape[1])
        num = 0
        Sum_error = 0
        while Sum_error < Threshold and num <= Img_tem.shape[0]*Img_tem.shape[1]: 
            i = random.randint(0, Img_win.shape[0]-1)
            j = random.randint(0, Img_win.shape[1]-1)  # 这里的随机选取无法保证不重复
            Sum_error = Sum_error + abs(Img_win[i,j]-E_win-Img_tem[i,j]+E_tem)
            num = num + 1
        return num

# Sequential Similiarity Detection Algorithm
def SSDA_reg(Img, Img_crop):
    Img_matrix = np.matrix(Img)
    Img_crop_matrix = np.matrix(Img_crop)
    Img_height, Img_width = Img_matrix.shape
    Img_crop_height, Img_crop_width = Img_crop_matrix.shape
    Threshold = 50
    SSDA_num_max = 0
    SSDA_num_max_i = 0
    SSDA_num_max_j = 0
    for i in range(Img_height-Img_crop_height+1):
        for j in range(Img_width-Img_crop_width+1):
            Img_win = Img_matrix[i:Img_crop_height+i, j:Img_crop_width+j]
            SSDA_num = SSDA_calc(Img_win, Img_crop_matrix, Threshold)
            if SSDA_num > SSDA_num_max:
                SSDA_num_max = SSDA_num
                SSDA_num_max_i = i
                SSDA_num_max_j = j
    return [SSDA_num_max_i, SSDA_num_max_j, SSDA_num_max]

# Partitioned Intensity Uniformity calculation
def PIU_calc(Img_win, Img_tem):
    if Img_win.shape != Img_tem.shape:
        print('子图和模板尺寸不匹配')
        return 0
    else:
        Img_height = Img_win.shape[0]
        Img_width = Img_win.shape[1]
        N = Img_height * Img_width
        PIU = 0
        tmp_win = np.zeros(256)
        for i in range(Img_height):
            for j in range(Img_width):
                value = Img_win[i, j]
                tmp_win[value] = tmp_win[value] + 1  
        tmp_tem = np.zeros(256)
        for i in range(Img_height):
            for j in range(Img_width):
                value = Img_tem[i, j]
                tmp_tem[value] = tmp_tem[value] + 1 
        for i in range(tmp_win.size):
            if tmp_win[i] == 0:
                PIU = PIU
            else:
                Img_w2t = Img_tem[Img_win==i]
                mean_w2t =sum(Img_w2t)/tmp_win[i]
                var_w2t = sum((Img_w2t-mean_w2t*np.ones(Img_w2t.size))**2)/tmp_win[i]
                PIU = PIU + tmp_win[i]*var_w2t / (N*mean_w2t)
        for i in range(tmp_tem.size):
            if tmp_tem[i] == 0:
                PIU = PIU
            else:
                Img_t2w = np.array(Img_win[Img_tem==i])
                mean_t2w  = sum(Img_t2w)/tmp_tem[i]
                var_t2w  = sum((Img_t2w-mean_t2w*np.ones(Img_t2w.size))**2)/tmp_tem[i]
                PIU = PIU + tmp_tem[i]*var_t2w / (N*mean_t2w)
        return PIU

# Partitioned Intensity Uniformity registration
def PIU_reg(Img, Img_crop):
    Img_matrix = np.array(Img)   # need to figure out the differences between np.array and np.matrix in numpy
    Img_crop_matrix = np.array(Img_crop)
    Img_height, Img_width = Img_matrix.shape
    Img_crop_height, Img_crop_width = Img_crop_matrix.shape
    PIU_value_min = 100000
    PIU_value_min_i = 0 
    PIU_value_min_j = 0
    for i in range(Img_height-Img_crop_height+1):
        for j in range(Img_width-Img_crop_width+1):
            Img_win = Img_matrix[i:Img_crop_height+i, j:Img_crop_width+j]
            PIU_value = PIU_calc(Img_win, Img_crop_matrix)
            if PIU_value < PIU_value_min:
                PIU_value_min = PIU_value
                PIU_value_min_i = i
                PIU_value_min_j = j
            if PIU_value_min == 0:
                break
    return [PIU_value_min_i, PIU_value_min_j, PIU_value_min]

if __name__=='__main__':
    Img = cv2.imread("F://study//python//python_programming//resource//test.jpg", flags=2)  # read the image in gray level
    Img_crop = cv2.imread('F://study//python//python_programming//resource//test_crop.jpg', flags=2)  # read the crop image as template
    #print(MAD_reg(Img, Img_crop))
    #print(SSDA_reg(Img, Img_crop))
    print(PIU_reg(Img, Img_crop))