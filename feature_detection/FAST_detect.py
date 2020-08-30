# this program is to use FAST to detect feature points
'''
一些记录：
有结果了，极大值抑制还不行，第二步检验不是连续判断
'''

import cv2
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
        i = 0
        while i <= Result_List_input.shape[0]-1:
            y, x = Result_List_input[i-1,0:2]
            position_max = np.array([y_max, x_max])
            position = np.array([y, x])
            if np.sqrt(np.sum((position_max - position)**2)) <= radius:
                Result_List_input = np.delete(Result_List_input, i-1, 0)
                Detect_Result_nms[int(y), int(x)] = 0
                i = i - 1 
            i = i + 1
    return Detect_Result_nms, Result_List_output

# 将识别到的角点可视化到原图像上
def Result_Display(Img, FASTResult):
    color_img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
    color_img[FASTResult==1] = [0, 0, 255]
    cv2.namedWindow("Img-FAST")
    cv2.imshow("Img-FAST", color_img)
    cv2.waitKey(5000) 
    cv2.destroyAllWindows()

if __name__=='__main__':
    Img = cv2.imread("F://study//python//python_programming//resource//test.jpg", 0) # read the image in gray level 
    Threshold = 50
    FASTResult, Result_List = FAST_detect(Img, Threshold)
    # FASTResult = Non_maximum_suppresion(FASTResult, 5)
    # print(Result_List)
    Result_Display(Img, FASTResult)
    Detect_Result_nms, Result_List_nms = Non_maximum_suppresion(FASTResult, Result_List, 3)
    Result_Display(Img, Detect_Result_nms)