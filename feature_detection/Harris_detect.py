# this program is to use Harris to detect the corners
'''
一些记录：
1.目前还没有极大值抑制，用常规方法求梯度Ix和Iy时结果都是负的，但用sobel求梯度时有正的，但需要很大的阈值，且结果不太像角点，好的用棋盘格试了一些果然不行

'''
import cv2
import numpy as np
import math

# 输入图像、邻域窗宽、阈值得到一个与图像同纬度的0-1矩阵，其中值为1的像素对应图像位置为Harris角点
def Harris_detect(Img, WinWidth, Threshold):
    Img_matrix = np.array(Img)
    Img_height, Img_width = Img_matrix.shape
    HarrisResult = np.zeros((Img_height, Img_width))
    # 整图梯度计算
    Iy, Ix = np.gradient(Img_matrix)
    '''
    # 用sobel算子求图像梯度
    sobelx = cv2.Sobel(Img_matrix, cv2.CV_64F, dx=1, dy=0)
    Ix = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(Img_matrix, cv2.CV_64F, dx=0, dy=1)
    Iy = cv2.convertScaleAbs(sobely)
    # 这里有问题，最终找到的是角点附近的点。感觉是因为sobel计算梯度是隔一个做差？
    '''
    '''
    # 普通方法计算图像梯度
    Ix = np.zeros(Img_matrix.shape)
    Iy = Ix
    for x in range(Img_matrix.shape[1]-1):
        for y in range(Img_matrix.shape[0]-1):
            Ix[y, x] = int(Img_matrix[y, x+1]) - int(Img_matrix[y, x])   #  图像像素值是ubyte类型，ubyte类型数据范围为0~255，若做运算出现负值或超出255，则会抛出异常
            Iy[y, x] = int(Img_matrix[y+1, x]) - int(Img_matrix[y, x])
    # 这种方法也不行，甚至找不到角点
    '''
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy
    offset = WinWidth//2
    # 目前该程序因为窗宽的影响没有外圈补充，故计算时没有考虑原图像窗宽宽度的外围像素
    for v in range(offset, Img_width-offset):
        for u in range(offset, Img_height-offset):
            Img_winxx = Ixx[u-offset:u+offset+1, v-offset:v+offset+1]
            Img_winyy = Iyy[u-offset:u+offset+1, v-offset:v+offset+1]
            Img_winxy = Ixy[u-offset:u+offset+1, v-offset:v+offset+1]
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
                HarrisResult[u, v] = 1
    return HarrisResult


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
    color_img[HarrisResult==1] = [0, 0, 255]
    cv2.namedWindow("Img-Harris")
    cv2.imshow("Img-Harris", color_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__=='__main__':
    Img = cv2.imread("F://study//python//python_programming//resource//checkerboard2.jpg", 0) # read the image in gray level 
    WinWidth = 5  # 邻域窗宽
    Threshold = 10000  # 算法阈值 
    Detect_Result = Harris_detect(Img, WinWidth, Threshold)
    #print(sum(sum(Detect_Result)))
    Result_Display(Img, Detect_Result)
    

    '''
    # 自带的Harris角点检测
    dst = cv2.cornerHarris(Img, 2, 3, 0.04)
    # 角点标记为红色
    Img_raw[dst > 0.01 * dst.max()] = [0, 255, 0]
    cv2.imshow('dst', Img_raw)
    cv2.waitKey(0)
    '''