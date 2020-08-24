# this program is to use FAST to detect feature points
'''
一些记录：
有结果了，目前还没有极大值抑制，第二步检验不是连续判断
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
                if delta_win.sum() > 13:
                    FASTResult[y, x] = 1
    return FASTResult

# 将识别到的角点可视化到原图像上
def Result_Display(Img, FASTResult):
    color_img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
    color_img[FASTResult==1] = [0, 0, 255]
    cv2.namedWindow("Img-FAST")
    cv2.imshow("Img-FAST", color_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__=='__main__':
    Img = cv2.imread("F://study//python//python_programming//resource//test.jpg", 0) # read the image in gray level 
    Threshold = 50
    FASTResult = FAST_detect(Img, Threshold)
    Result_Display(Img, FASTResult)