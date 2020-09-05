# this program is to use FAST to detect feature points
'''
一些记录：
有结果了，极大值抑制看起来还行，第二步检验不是连续判断
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np

# FAST特征检测
def FAST_detect(Img, Threshold):
    mask = np.array([[0,0,1,1,1,0,0], 
                    [0,1,0,0,0,1,0], 
                    [1,0,0,0,0,0,1], 
                    [1,0,0,0,0,0,1], 
                    [1,0,0,0,0,0,1], 
                    [0,1,0,0,0,1,0], 
                    [0,0,1,1,1,0,0]])
    Img_matrix = np.array(Img)
    FASTResult = np.zeros(Img_matrix.shape)
    Result_List = []
    for y in range(3, Img_matrix.shape[0]-4):
        for x in range(3, Img_matrix.shape[1]-4):
            centerValue = int(Img_matrix[y, x])
            delta1 = abs(int(Img_matrix[y-3, x]) - centerValue)
            delta5 = abs(int(Img_matrix[y, x+3]) - centerValue)
            delta9 = abs(int(Img_matrix[y+3, x]) - centerValue)
            delta13 = abs(int(Img_matrix[y, x-3]) - centerValue)
            delta = np.array([delta1, delta5, delta9, delta13]) > Threshold
            #print((y, x), delta)
            if delta.sum() >= 3:
                Img_win = mask * Img_matrix[y-3:y+4, x-3:x+4]
                delta_win = (abs(Img_win - centerValue*mask) > Threshold) # 这里没有连续判断阈值
                #print(delta_win)
                if delta_win.sum() > 12:
                    score = sum(sum(abs(Img_win - centerValue*mask) ** 2))
                    FASTResult[y, x] = 1
                    Result_List.append([y, x, score])
    Result_List = np.array(Result_List)
    return FASTResult, Result_List

# 非极大值抑制
# https://zhuanlan.zhihu.com/p/78504109
def Non_maximum_suppresion(Detect_Result, Result_List, radius):
    Result_List_input = Result_List[np.argsort(-Result_List[:,2])] # 根据R值由高到低排序
    Detect_Result_nms = Detect_Result
    Result_List_output = []
    while Result_List_input.size > 0:
        # 将最大的R值对应坐标挪到输出列表中
        y_max, x_max, R_max = Result_List_input[0,:]
        Result_List_output.append([y_max, x_max, R_max])
        Result_List_input = np.delete(Result_List_input, 0, 0) 
        # 剩下的R逐个与目前最大判断范围,若在范围内则被抑制(去除)
        i = 1
        while i <= Result_List_input.shape[0]:
            y, x = Result_List_input[i-1,0:2]
            position_max = np.array([y_max, x_max])
            position = np.array([y, x])
            if np.sqrt(np.sum((position_max - position)**2)) <= radius:
                Result_List_input = np.delete(Result_List_input, i-1, 0)
                Detect_Result_nms[int(y), int(x)] = 0
                i = i - 1 
            i = i + 1
    Result_List_output = np.array(Result_List_output)
    return Detect_Result_nms, Result_List_output

# BRISK特征描述
def BRISK_description(Img, point_list, winwidth):
    BRISK_result = []
    mean = [0, 0]
    cov = [[winwidth**2/25, 0], [0, winwidth**2/25]]
    # 高斯平滑降噪
    Img_gauss = cv2.GaussianBlur(Img, (9, 9), sigmaX=2, sigmaY=2)
    for i in range(point_list.shape[0]):
        y, x = point_list[i, 0:2]
        Img_matrix = np.array(Img_gauss)
        encoder = []
        # 每个描述子有256位编码
        for i in range(256):
            # 由(0,winwidth/25)高斯分布随机取两个点
            x1, y1 = np.random.multivariate_normal(mean, cov, 1).T
            x2, y2 = np.random.multivariate_normal(mean, cov, 1).T
            if Img_matrix[3+int(y1), 3+int(x1)] >= Img_matrix[3+int(y2), 3+int(x2)]:
                encoder.append(1)
            else:
                encoder.append(0)
        BRISK_result.append(encoder)
    BRISK_result = np.array(BRISK_result, dtype='uint8') # 特征匹配函数必须要uint8
    return BRISK_result

# 将识别到的角点可视化到原图像上
def Result_Display(Img, FASTResult):
    color_img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
    color_img[FASTResult==1] = [0, 0, 255]
    cv2.namedWindow("Img-FAST")
    cv2.imshow("Img-FAST", color_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__=='__main__':
    Img_T = cv2.imread("F://study//python//python_programming//resource//test.jpg", 0) # read the image in gray level 
    Img_m = cv2.imread("F://study//python//python_programming//resource//test_rotated.jpg", 0)
    Threshold = 50
    #FASTResult, Result_List = FAST_detect(Img, Threshold)
    #Detect_Result_nms, Result_List_nms = Non_maximum_suppresion(FASTResult, Result_List, 3)
    #BRISK_result = BRISK_description(Img, Result_List_nms, 7)
    #print(BRISK_result)
    #Result_Display(Img, Detect_Result_nms)
    print("Detect begin!")
    # detect the template image
    #orb = cv2.ORB_create()
    #kp1, des1 = orb.detectAndCompute(Img_T,None)
    #print(des1)
    FASTResult_T, Result_List_T = FAST_detect(Img_T, Threshold)
    Detect_Result_nms_T, Result_List_nms_T = Non_maximum_suppresion(FASTResult_T, Result_List_T, 3)
    BRISK_result_T = BRISK_description(Img_T, Result_List_nms_T, 7)
    # detect the moving image
    #kp2, des2 = orb.detectAndCompute(Img_m,None)
    FASTResult_m, Result_List_m = FAST_detect(Img_m, Threshold)
    Detect_Result_nms_m, Result_List_nms_m = Non_maximum_suppresion(FASTResult_m, Result_List_m, 3)
    BRISK_result_m = BRISK_description(Img_m, Result_List_nms_m, 7)
    print("Detect OK!")
    # try to match two images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #建立匹配关系
    #matches = bf.match(des1,des2)
    #matches = sorted(matches,key=lambda x:x.distance)
    #result= cv2.drawMatches(Img_T, kp1, Img_m, kp2, matches[:40],None,flags=2)
    matches = bf.match(BRISK_result_T, BRISK_result_m)
    matches = sorted(matches,key=lambda x:x.distance)
    # 将坐标点集转为关键点类型
    kp1 = [cv2.KeyPoint(Result_List_nms_T[i][0], Result_List_nms_T[i][1], 1) for i in range(Result_List_nms_T.shape[0])]
    kp2 = [cv2.KeyPoint(Result_List_nms_m[i][0], Result_List_nms_m[i][1], 1) for i in range(Result_List_nms_m.shape[0])]
    result= cv2.drawMatches(Img_T, kp1, Img_m, kp2, matches[:40],None,flags=2) #画出匹配关系
    plt.imshow(result),plt.show() #matplotlib描绘出来
