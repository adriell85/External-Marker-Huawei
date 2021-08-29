import cv2
import matplotlib.pyplot as ppl
import numpy as np
from skimage.segmentation import flood, flood_fill

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

kernel = np.ones((7,7),np.uint8)

closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

canny = cv2.Canny(closing,10,50)
black = np.zeros((width,height,3),dtype=np.uint8)

black = cv2.resize(black,dim,interpolation=cv2.INTER_AREA)



contours,hierarchy =cv2.findContours(binary_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(black,contours,-1,(0,255,0),2)



bl = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)

rows,cols = bl.shape[:2]

for row in range(rows):
    for col in range(cols):
        if (row<200 and col < cols):
            bl[row][col] = 0
        elif(row<rows and col<50):
            bl[row][col] = 0
        else:
            bl[row][col] = bl[row][col]

light_coat = flood_fill(bl, (155, 150), 255, tolerance=10)

light_coat = np.uint8(np.where(light_coat==0,255,0))

light_coat = cv2.morphologyEx(light_coat, cv2.MORPH_OPEN, kernel)



contours,hierarchy =cv2.findContours(light_coat,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

max_value_contour = []
for contour in contours:
    if len(contour) > len(max_value_contour):
        max_value_contour = contour


rect = cv2.minAreaRect(max_value_contour)
box = cv2.boxPoints(rect)
box = box.astype('int')

img_copy = img.copy()
img_box_2 = cv2.drawContours(black, contours = [box],
                             contourIdx = -1,  color = (0, 0, 255), thickness = 2)

cv2.imshow('img',black)
cv2.waitKey(0)
