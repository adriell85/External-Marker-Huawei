from numba import njit
import cv2
from pynput.mouse import Controller
import numpy as np


def kmeans_segmentation(img, K):

    import numpy as np
    import cv2

    Z = img.reshape((-1, 2))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2

# @njit
# def regionGrowing(img, object, xd, yd):
#
#     rows, cols = img.shape[:2]
#     inicio=0
#     fim=1
#     while(inicio!=fim):
#         inicio=fim
#         fim=0
#         for row in range(1,rows-1):
#             for col in range(1,cols-1):
#                 if object[row,col]==255:
#                     for i in range(-1,2):
#                         for j in range(-1,2):
#                             if (img[row+i,col+j]<(img[xd,yd])+50)&(img[row+i,col+j]>(img[xd,yd])-50):
#                                 object[row+i,col+j]=255
#                                 # colors[row+i,col+j]=img[row+i,col+j]
#                                 fim+=1
#     return object
#
# # mouse = Controller()
# # mouse.position
#
# point_matrix = np.zeros((2, 2), int)
#
#
#
# def mousePoints(event, x, y, flags, params):
#     global point_matrix
#     if event == cv2.EVENT_LBUTTONDOWN:
#         point_matrix = x, y
#         img2 = regionGrowing(img, np.zeros((img.shape[1], img.shape[0])), point_matrix[1], point_matrix[0])
#         cv2.imshow('test', img2)
#
#
#
img = cv2.imread('fundo_preto.jpg',1)
scale_percent = 20  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)


img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
orig = img
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# img = kmeans_segmentation(img,3)


# cv2.imshow('img',img)
#
# cv2.setMouseCallback("img",mousePoints)

# cv2.waitKey()




def region_growing_gray(img):
    rows, cols = img.shape[:2]
    objeto = np.zeros((rows, cols), np.uint8)
    colors = np.zeros((rows, cols), np.uint8)
    rows_new = int(rows / 1)
    cols_new = int(cols / 1)
    # img = cv2.resize(img, (rows_new, cols_new))
    # objeto = cv2.resize(objeto, (rows_new, cols_new))
    # colors = cv2.resize(colors, (rows_new, cols_new))



    def Mouse_click(event, y, x, flags, param):
        if (event == cv2.EVENT_LBUTTONDOWN):
            global xd
            global yd
            xd = x
            yd = y
            # objeto=region_growing_binary(img)
            objeto[xd, yd] = 255
            print(xd)
            print(yd)



    cv2.imshow('imagem', img)
    cv2.namedWindow('imagem', 1)
    cv2.setMouseCallback('imagem', Mouse_click)
    # objeto[xd, yd] = 255
    cv2.waitKey(0)

    # return cv2.resize(iterations_growing_gray(img,objeto,colors),(rows,cols))
    return iterations_growing_gray(img,objeto,colors)

@njit
def iterations_growing_gray(img,objeto,colors):

    rows,cols=img.shape[:2]
    inicio=0
    fim=1
    while(inicio!=fim):
        inicio=fim
        fim=0
        for row in range(1,rows-1):
            for col in range(1,cols-1):
                if objeto[row,col]==255:
                    for i in range(-1,2):
                        for j in range(-1,2):
                            if (img[row+i,col+j]<(img[xd,yd])+80)&(img[row+i,col+j]>(img[xd,yd])-80):
                                objeto[row+i,col+j]=255
                                colors[row+i,col+j]=img[row+i,col+j]
                                fim+=1
                    # cv2.imshow('colors',colors)
                    # cv2.waitKey(1)

    return colors

img = region_growing_gray(img)

gray = cv2.bilateralFilter(img, 11, 17, 17)
Blur=cv2.GaussianBlur(gray,(5,5),1)
edged = cv2.Canny(Blur, 10, 250)
contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(orig, contours, -1, (0, 255, 0), 3)

black = np.zeros((width,height,3))
black = cv2.resize(black, dim, interpolation=cv2.INTER_AREA)
cv2.drawContours(black, contours, -1, (0, 255, 0), 3)

cv2.imshow('black',black)

cv2.imshow('w',orig)
cv2.waitKey(0)