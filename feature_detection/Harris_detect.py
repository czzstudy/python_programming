# this program is to use Harris to detect the corners
'''
一些记录：
非极大值抑制看起来还行，sobel求梯度的方法不太行，numpy求梯度的方法会得到角点旁边很多点，目前用常规方法求出来的效果较好

'''
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# 输入图像、邻域窗宽、阈值得到一个与图像同纬度的0-1矩阵，其中值为1的像素对应图像位置为Harris角点
def Harris_detect(Img, WinWidth, Threshold):
    Img_matrix = np.array(Img)
    Img_height, Img_width = Img_matrix.shape
    HarrisResult = np.zeros((Img_height, Img_width))
    # 整图梯度计算
    # Iy, Ix = np.gradient(Img_matrix)
    '''
    # 用sobel算子求图像梯度
    sobelx = cv2.Sobel(Img_matrix, cv2.CV_64F, dx=1, dy=0)
    Ix = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(Img_matrix, cv2.CV_64F, dx=0, dy=1)
    Iy = cv2.convertScaleAbs(sobely)
    # 这里有问题，最终找到的是角点附近的点。感觉是因为sobel计算梯度是隔一个做差？
    '''
    
    # 普通方法计算图像梯度
    Ix = np.zeros(Img_matrix.shape)
    Iy = Ix.copy()   # 直接=会导致Ix随Iy变化
    for x in range(Img_matrix.shape[1]-1):
        for y in range(Img_matrix.shape[0]-1):
            Ix[y, x] = int(Img_matrix[y, x+1]) - int(Img_matrix[y, x])   #  图像像素值是ubyte类型，ubyte类型数据范围为0~255，若做运算出现负值或超出255，则会抛出异常
            Iy[y, x] = int(Img_matrix[y+1, x]) - int(Img_matrix[y, x])
    
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy
    offset = WinWidth//2
    Result_List = []
    # 目前该程序因为窗宽的影响没有外圈补充，故计算时没有考虑原图像窗宽宽度的外围像素
    for x in range(offset, Img_width-offset):
        for y in range(offset, Img_height-offset):
            Img_winxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            Img_winyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Img_winxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            # 计算得到每个像素点对应的4*4椭圆特征矩阵
            Gauss_matrix = Gauss_Mcalc(Img_winxx.shape[0])
            feature_matrix = np.zeros((2, 2))
            feature_matrix[0, 0] = sum(sum(Img_winxx*Gauss_matrix))
            feature_matrix[1, 1] = sum(sum(Img_winyy*Gauss_matrix))
            feature_matrix[0, 1] = sum(sum(Img_winxy*Gauss_matrix))
            feature_matrix[1, 0] = sum(sum(Img_winxy*Gauss_matrix))
            R = Harris_Rcalc(feature_matrix)
            # print(R)
            if R > Threshold:
                HarrisResult[y, x] = R
                Result_List.append([y, x, R])
    Result_List = np.array(Result_List)
    return HarrisResult, Result_List


# 由窗宽得到一个符合高斯分布的矩阵
def Gauss_Mcalc(Win_width):
    centerx = Win_width//2
    centery = centerx
    Gauss_matrix = np.zeros((Win_width, Win_width))
    for i in range(Win_width):
        for j in range(Win_width):
            Gauss_matrix[i, j] = Gauss_Function(i-centerx, j-centery, 0, 1)  # 高斯分布，均值为0，标准差为1
    return Gauss_matrix

# 二维高斯分布函数
def Gauss_Function(x, y, mean, var):
    # return 1
    return math.exp(-(x**2+y**2)/(2*(var**2)))/(2*math.pi*(var**2))


# 根据4*4椭圆特征矩阵计算Harris需要的特征值
def Harris_Rcalc(feature_matrix):
    R = np.linalg.det(feature_matrix) - 0.04*((np.trace(feature_matrix))**2)
    return R

# 将识别到的角点可视化到原图像上
def Result_Display(Img, HarrisResult):
    color_img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
    color_img[HarrisResult>0] = [0, 0, 255]
    cv2.namedWindow("Img-Harris")
    cv2.imshow("Img-Harris", color_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

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

# BRIEF特征描述
def BRIEF_description(Img, point_list, winwidth):
    BRIEF_result = []
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
        BRIEF_result.append(encoder)
    BRIEF_result = np.array(BRIEF_result, dtype='uint8') # 特征匹配函数必须要uint8
    return BRIEF_result

if __name__=='__main__':
    Img_T = cv2.imread("F://study//python//python_programming//resource//test.jpg", 0) # read the image in gray level 
    Img_m = cv2.imread("F://study//python//python_programming//resource//test_rotated.jpg", 0)
    WinWidth = 5  # 邻域窗宽
    Threshold = 10000  # 算法阈值 
    #Detect_Result, Result_List = Harris_detect(Img, WinWidth, Threshold)
    #Detect_Result_nms, Result_List_nms = Non_maximum_suppresion(Detect_Result, Result_List, 3)
    #BRIEF_result = BRIEF_description(Img, Result_List_nms, 7)
    #Result_Display(Img, Detect_Result_nms)

    print("Detect begin!")
    # detect the template image
    #orb = cv2.ORB_create()
    #kp1, des1 = orb.detectAndCompute(Img_T,None)
    #print(des1)
    HarrisResult_T, Result_List_T = Harris_detect(Img_T, WinWidth, Threshold)
    Detect_Result_nms_T, Result_List_nms_T = Non_maximum_suppresion(HarrisResult_T, Result_List_T, 3)
    BRIEF_result_T = BRIEF_description(Img_T, Result_List_nms_T, 7)
    # detect the moving image
    #kp2, des2 = orb.detectAndCompute(Img_m,None)
    HarrisResult_m, Result_List_m = Harris_detect(Img_m, WinWidth, Threshold)
    Detect_Result_nms_m, Result_List_nms_m = Non_maximum_suppresion(HarrisResult_m, Result_List_m, 3)
    BRIEF_result_m = BRIEF_description(Img_m, Result_List_nms_m, 7)
    print("Detect OK!")
    # try to match two images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #建立匹配关系
    #matches = bf.match(des1,des2)
    #matches = sorted(matches,key=lambda x:x.distance)
    #result= cv2.drawMatches(Img_T, kp1, Img_m, kp2, matches[:40],None,flags=2)
    matches = bf.match(BRIEF_result_T, BRIEF_result_m)
    matches = sorted(matches,key=lambda x:x.distance)
    # 将坐标点集转为关键点类型
    kp1 = [cv2.KeyPoint(Result_List_nms_T[i][0], Result_List_nms_T[i][1], 1) for i in range(Result_List_nms_T.shape[0])]
    kp2 = [cv2.KeyPoint(Result_List_nms_m[i][0], Result_List_nms_m[i][1], 1) for i in range(Result_List_nms_m.shape[0])]
    result= cv2.drawMatches(Img_T, kp1, Img_m, kp2, matches[:40],None,flags=2) #画出匹配关系
    plt.imshow(result),plt.show() #matplotlib描绘出来


    '''
    # 自带的Harris角点检测
    dst = cv2.cornerHarris(Img, 2, 3, 0.04)
    # 角点标记为红色
    Img_raw[dst > 0.01 * dst.max()] = [0, 255, 0]
    cv2.imshow('dst', Img_raw)
    cv2.waitKey(0)
    '''