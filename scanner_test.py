import cv2
import matplotlib.pyplot as ppl
import numpy as np
from skimage.segmentation import flood, flood_fill

# photos = ['scan.jpg','fundo_madeira.jpg','fundo_preto.jpg','fundo_teclado.jpg']

# for i in photos:
#
#
#     img = cv2.imread(i)
#     # img = cv2.imread('fundo_madeira.jpg')
#     # img = cv2.imread('fundo_preto.jpg')
#     # img = cv2.imread('fundo_teclado.jpg')
#
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#     print('Original Dimensions : ', img.shape)
#
#     scale_percent = 20  # percent of original size
#     width = int(img.shape[1] * scale_percent / 100)
#     height = int(img.shape[0] * scale_percent / 100)
#     dim = (width, height)
#
#     # resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
#
#
#
#     # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8,ltype=cv2.CV_32S)
#     # imgplot = ppl.imshow(img)
#     # ppl.show()
#     #
#     # gray = np.where(gray == 255,255,0)
#     # gray = np.uint8(gray)
#
#     # edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#     #                                       cv2.THRESH_BINARY, 199, 5)
#
#     gray = cv2.bilateralFilter(gray, 11, 17, 17)
#     edged = cv2.Canny(gray, 150, 200)
#     black = np.zeros((width, height,3),dtype=np.uint8)
#
#
#
# # ====================================================================================================================
# #     _, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
# #     _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours, hierarchy = cv2.findContours(edged,
#                                            cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# # ====================================================================================================================
#
#     img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#
#
#
#     cv2.imshow("Resized image", img)
#     # cv2.imwrite('{}.jpg'.format(i),img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


color = cv2.imread('scan.jpg',1)
img = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
scale_percent = 40  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
color = cv2.resize(color,dim,interpolation=cv2.INTER_AREA)
binary_img = np.where(img > 250,255,0)
binary_img = np.uint8(binary_img)



Blur=cv2.GaussianBlur(binary_img,(5,5),1) #apply blur to roi
Canny=cv2.Canny(Blur,10,50) #apply canny to roi

#Find my contours
contours =cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

black = np.zeros((width,height,3))
black = cv2.resize(black,dim,interpolation=cv2.INTER_AREA)

#Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
cntrRect = []
for i in contours:
    # if cv2.contourArea(i) > 15:
            epsilon = 0.05*cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,epsilon,True)
            if len(approx) == 4:
                cv2.drawContours(color,cntrRect,-1,(0,255,0),2)
                cv2.drawContours(black, cntrRect, -1, (0, 255, 0), 2)

                cntrRect.append(approx)
                # x, y, w, h = cv2.boundingRect(i)
                # # Draw the rectangle
                # cv2.rectangle(black, (x, y), (x + w, y + h), (255, 255, 0), 1)

                cv2.imshow('Roi Rect ONLY', black)
#===================================
# Solução para capturar os 4 pontos
# func_image = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)
#
# nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8,ltype=cv2.CV_32S)
# imgplot = ppl.imshow(output)
# ppl.show()
#==================================
kernel = np.asarray([[0,1,0],
          [1,1,1],
          [0,1,0]],np.uint8)
# kernel = np.ones((3,3),np.uint8)
Canny = np.where(Canny > 200,255,0)
Canny = np.uint8(Canny)
Canny = cv2.dilate(Canny,kernel)
light_coat = flood_fill(Canny, (155, 150), 255, tolerance=10)
light_coat = np.uint8(light_coat)

cont, hierarchy = cv2.findContours(light_coat,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(light_coat, cont, -1, (0, 255, 0), 3)

x, y, w, h = cv2.boundingRect(light_coat)
                # Draw the rectangle
cv2.rectangle(light_coat, (x, y), (x + w, y + h), (255, 255, 0), 1)

cv2.imshow('image',light_coat)
cv2.waitKey()

