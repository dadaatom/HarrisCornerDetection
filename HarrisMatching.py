import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random

def harris(image, k, thresh, windowSize = 5, localSize = 16):
    keypoints = []

    Ix, Iy = np.gradient(image)

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Iy * Ix

    Ixx = cv2.GaussianBlur(Ixx, (3, 3), 1)
    Iyy = cv2.GaussianBlur(Iyy, (3, 3), 1)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), 1)

    display('Ix', Ix)
    display('Iy', Iy)
    display('Ixy', Ixy)

    responses = np.zeros((image.shape[0], image.shape[1]))

    offset = int(windowSize / 2)
    for y in range(offset, image.shape[0] - offset):
        for x in range(offset, image.shape[1] - offset):

            x0 = x - offset
            y0 = y - offset
            y1 = y + offset + 1
            x1 = x + offset + 1

            Sumx = Ixx[y0: y1, x0: x1].sum()
            Sumxy = Ixy[y0: y1, x0: x1].sum()
            Sumy = Iyy[y0: y1, x0: x1].sum()

            determinant = (Sumx * Sumy) - (Sumxy ** 2)
            responses[y, x] = determinant - k * ((Sumx + Sumy) ** 2)

    responses[np.where(responses <= thresh)] = 0

    for y in range(int(image.shape[0]/localSize)):
        for x in range(int(image.shape[1]/localSize)):

            ySize = localSize
            xSize = localSize

            if y == int(image.shape[0] / localSize) - 1:
                ySize = image.shape[0] - y * localSize
            if x == int(image.shape[1] / localSize) - 1:
                xSize = image.shape[1] - x * localSize

            localChunk = responses[y * localSize: y * localSize + ySize, x * localSize: x * localSize + xSize]

            maxY = 0
            maxX = 0
            for localY in range(ySize):
                for localX in range(xSize):
                    if localChunk[localY, localX] > thresh and localChunk[localY, localX] > localChunk[maxY, maxX]:
                        maxY = localY
                        maxX = localX

            if localChunk[maxY, maxX] > 0:
                keypoints.append(cv2.KeyPoint(x*localSize + maxX, y*localSize + maxY, localChunk[maxY, maxX]))

    return keypoints, responses

def display(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)


# ===================== IMAGES ===================== #

hough1Path = "Images/hough1.png"
hough2Path = "Images/hough2.png"


# ===================== Execution ===================== #
if __name__ == '__main__':
    imagePath = hough1Path

    image = cv2.imread(imagePath)
    image.astype('uint8')
    
    grayscaleImage = cv2.imread(imagePath,0).astype('uint8')
    grayscaleImage.astype('uint8')

    display('original', image)

    print('Computing...')

    keypoints, responses = harris(grayscaleImage, .04, 100000)

    display('responses', responses)
    display('harris', cv2.drawKeypoints(image, keypoints, None))

    print('Done!')

    cv2.waitKey(0)
