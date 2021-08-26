# !/usr/bin/python3
# coding: utf-8

# Created by iagsoncarlos on Tuesday, October 16, 2019.
# Copyright (c) 2019 Iágson Carlos Lima Silva. All rights reserved.

# ------------------------------------------------------------------------------------------------------------------------
# Import required libraries

from skimage.segmentation import clear_border
from os.path import abspath, join, dirname
from mls_parzen import mls_parzen, conv2
from scipy.signal import convolve2d
from time import time

import matplotlib.pyplot as plt
import numpy as np
import skimage
import sys
import cv2
from numba import njit

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from data_information import dcm_information as di
from kmeans_clustering import *


def normalize_image(image):
    image = (image - image.min()) / (image.max() - image.min())
    result = (image * 255).astype(np.uint8)
    return result

def largest_component(image, n_comp):

    image = np.asarray(image, np.uint8)

    connectivity = 8

    output2 = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_16SC1)
    labels2 = output2[1]
    stats2 = output2[2]
    # Pegar as componentes do maior para menor
    # Sum of each component = output[2]
    indice = np.argsort(output2[2][:, -1])[::-1]
    
    lung1 = np.zeros(image.shape, image.dtype)
    lung2 = np.zeros(image.shape, image.dtype)

    try:
        largecomponent21 = 1 + stats2[1:, cv2.CC_STAT_AREA].argmax()
        stats2[largecomponent21, cv2.CC_STAT_AREA] = largecomponent21

        largecomponent22 = 1 + stats2[1:, cv2.CC_STAT_AREA].argmax()
        stats2[largecomponent21, cv2.CC_STAT_AREA] = largecomponent22

    except ValueError:
        return np.zeros(image.shape)

    lung1[labels2 == largecomponent21] = 255
    lung2[labels2 == largecomponent22] = 255

    # Se if == 1: Retorna a maior componente em termos de area
    # Se if == 2: Retorna a segunda maior componentes em termos de area
    # Se if == 3: Retorna a soma das duas maiores componentes em termos de area

    if n_comp == 1:
        return np.asarray(lung1, np.uint8)
    if n_comp == 2:
        return np.asarray(lung2, np.uint8)
    if n_comp == 3:
        return np.asarray((lung1 + lung2), np.uint8)


    cv2.imshow(str(__file__), image)
    cv2.waitKey(0)

def floodfill_image(im_in, n):
    # n = binary image threshold

    im_th = normalize_image(im_in)
    th, im_th = cv2.threshold(im_in, n, 127, cv2.THRESH_BINARY_INV);

    # im_th = normalize_image(im_th)
    # cv2.imshow('im_th', ~im_th)

    im_th = ~im_in
    # im_th = normalize_image(im_th)
    im_th = largest_component(~im_th, 3)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    fill_image = im_th | im_floodfill_inv

    return fill_image 
    # return im_floodfill_inv 

def geodesic_dilate(src, n, r, paddr, paddl, k, flag=False):

    kernel = np.ones((n, n), np.uint8)

    w, h = src.shape
    # r = 20
    # paddr = 40
    # paddl = 40

    marker = src.copy()
    marker[0:int(h/2)-r,:] = 0
    marker[int(h/2)+r:h,:] = 0
    marker[:,0:paddr] = 0
    marker[:,w-paddl:w] = 0

    # k = 150
    # ee = np.array([[1, 1, 1],[1, 1, 1], [1, 1, 1]])
    index = 0
    while index < k:
        marker = cv2.dilate(marker, kernel)
        cv2.multiply(marker, src, marker)
        index =  index + 1

    if flag == True:
        cv2.imshow('Geodesic Dilate', marker)
        cv2.waitKey(0)

    return marker

@njit
def regionGrowing(img, object, xd, yd):

    rows, cols = img.shape[:2]
    inicio=0
    fim=1
    while(inicio!=fim):
        inicio=fim
        fim=0
        for row in range(1,rows-1):
            for col in range(1,cols-1):
                if object[row,col]==255:
                    for i in range(-1,2):
                        for j in range(-1,2):
                            if (img[row+i,col+j]<(img[xd,yd])+50)&(img[row+i,col+j]>(img[xd,yd])-50):
                                object[row+i,col+j]=255
                                # colors[row+i,col+j]=img[row+i,col+j]
                                fim+=1
    return object

def centroidImage(image):
    centerImage = cv2.moments(image)
    cY = int(centerImage["m10"] / centerImage["m00"])
    cX = int(centerImage["m01"] / centerImage["m00"])

    obj = (int(cX), int(cY))

    return obj, int(cX), int(cY)   


if __name__ == "__main__":
    
    for r in range(1, 2):
        # print("Loop", r)
        average_time = []

        for z in range(1, 37):
            # plt.close()
            # Upload original image
            # imgOrig = di.load_dcm("../datasets/ImagensTC_Pulmao/{}.dcm".format(z))
            rootImageRGB = cv2.imread("/home/iagsoncarlos/Downloads/level-set-parzen/dataset/Grayscale/{}.png".format(z), 1)
            rootImageGray = cv2.imread("/home/iagsoncarlos/Downloads/level-set-parzen/dataset/Grayscale/{}.png".format(z), 0)

            # rootImageGray = (rootImageRGB, cv2.COLOR_BGR2GRAY)
            rootImageGray = cv2.resize(rootImageGray, (256, 256))

            imgOrig = kmeans_segmentation(rootImageRGB, 2)
            # imgOrig = cv2.medianBlur(rootImageRGB, 3)

            imgOrig = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

            # cv2.imshow('testing', rootImageGray)
            # cv2.waitKey(0)

            imgLo = cv2.imread("/home/iagsoncarlos/Downloads/level-set-parzen/dataset/MASK/{}.jpg".format(z), 0)
            imgLo = cv2.medianBlur(imgLo, 3)
            imgLo = cv2.resize(imgLo, (256, 256))

            # Resize image
            imgOrig = cv2.resize(imgOrig, (256, 256))

            # Normalize the image
            img_original = np.copy(imgOrig)
            # atemp = np.multiply(2.5, np.subtract(np.double(np.double(imgOrig)), 1024))
            # img = di.m_uint8(atemp)
            # imgOrig = imgOrig - 1024
            # imgOrig = imgOrig

            initial_time = time()


            kernel = np.ones((3, 3),np.uint8)
            # imgOrig.min()/2 == -1536.0
            # retval, var = cv2.threshold(imgOrig, -1536.0, 20, cv2.THRESH_BINARY_INV)
            retval, var = cv2.threshold(imgOrig, 127, 20, cv2.THRESH_BINARY_INV)

            # # Clear objects connected to the label image border.
            # var = geodesic_dilate(var, 3, 20, 40, 40, 150, False)

            centroindTemp, xd, yd = centroidImage(rootImageGray)
            imgShape = np.zeros((rootImageGray.shape), np.uint8)
            imgShape[xd, yd] = 255

            var = regionGrowing(rootImageGray, imgShape, xd, yd)

            # cv2.imshow('testing-debug', var)
            # cv2.waitKey(0)

            
            connectivity = 8
            output = cv2.connectedComponentsWithStats(np.asarray(var, np.uint8), connectivity, cv2.CV_8U)
            # Sum of each component = output[2]
            indice = np.argsort(output[2][:, -1])[::-1]

            # The second biggest component is the AVC
            # Label matrix = output[1]
            lung_component = np.where(output[1] == indice[0], 0, 1)
            lung_component = largest_component(lung_component, 3)

            # Morphology
            lung_component = cv2.morphologyEx(lung_component, cv2.MORPH_CLOSE, kernel)

            lung_component = floodfill_image(lung_component, 127)
            lung_component = np.where(lung_component < 1, 0, 1)
            shapes = di.m_uint8(imgOrig)
            lung_component = np.where(lung_component > 0, 1, 0)

            mask_image = np.ones(img_original.shape, np.uint8)

            # Fast Level Set Parzen
            X = np.asarray(rootImageGray, np.int16)
            mask = np.asarray(mask_image, np.uint8)
            Lo = cv2.erode(np.asarray((~imgLo), np.float64), kernel, iterations=4)
            dt = 0.9
            N = 20
            shapes = img_original.shape

            # time input (Parzen)
            phi, img, Psi, stop2_div = mls_parzen(X, mask, dt, N, Lo, shapes, False)
            # time output (Parzen)

            # Counting finish time
            final_time = time()

            total_time = (final_time - initial_time)
            print('[LOOP {} | TOTAL TIME: {:.5f}]'.format(z, total_time))
            average_time.append(total_time)

            Psi = cv2.morphologyEx(Psi, cv2.MORPH_CLOSE, kernel)
            Psi = cv2.resize(Psi, (512, 512))
            
            if Psi.any() > 0:
                cv2.imwrite("/home/iagsoncarlos/Downloads/level-set-parzen/results/{}.png".format(z), Psi)

            # print(z)

    # print('\n[AVERAGE TIME: {0}', np.mean(average_time))

    file = open("/home/iagsoncarlos/Downloads/level-set-parzen/results/time.txt", "w+")
    file.write('TOTAL TIME: \n{0}'.format(str(average_time)))
    file.write('\n\nAVERAGE TIME ~ Standard Deviation: {0}'.format(str(np.mean(average_time)) + ' ± {0}'.format(str(np.std(average_time)))))
    file.close()
