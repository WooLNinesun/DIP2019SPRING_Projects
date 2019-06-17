#!/usr/bin/env python
# coding: utf-8

import os, sys

import numpy as np
import cv2
from matplotlib import pyplot as plt

def showImage(image, title="", cmap=None):
    plt.figure(dpi=188)
    plt.imshow(image, cmap=None, vmin=0, vmax=255)
    plt.axis("off")
    plt.title(title)
    plt.show()


cvtMatrices = {}

cvtMatrices['rgb2xyz'] = np.array([[0.5141, 0.3239, 0.1604],
                                   [0.2651, 0.6702, 0.0641],
                                   [0.0241, 0.1228, 0.8444]])
cvtMatrices['xyz2rgb'] = np.linalg.inv(cvtMatrices['rgb2xyz'])

cvtMatrices['xyz2lms'] = np.array([[0.3897, 0.6890, -0.0787],
                                   [-0.2298, 1.1834, 0.0464],
                                   [0.0000, 0.0000, 1.0000]])
cvtMatrices['lms2xyz'] = np.linalg.inv(cvtMatrices['xyz2lms'])


cvtMatrices['rgb2lms'] = np.dot(cvtMatrices['xyz2lms'], cvtMatrices['rgb2xyz'])
cvtMatrices['lms2rgb'] = np.linalg.inv(cvtMatrices['rgb2lms'])


# LMS2LAB_Matrix1 = np.array([[1/np.sqrt(3), 0.0000, 0.0000],
#                             [0.0000, 1/np.sqrt(6), 0.0000],
#                             [0.0000, 0.0000, 1/np.sqrt(2)]])
# LMS2LAB_Matrix2 = np.array([[1.0000, 1.0000, 1.0000],
#                             [1.0000, 1.0000, -2.0000],
#                             [1.0000, -1.0000, 0.0000]])
# LMS2LAB_Matrix = np.dot(LMS2LAB_Matrix1, LMS2LAB_Matrix2)
cvtMatrices['lms2lab'] = np.array([[0.57735027,  0.57735027,  0.57735027],
                                   [0.40824829,  0.40824829, - 0.81649658],
                                   [0.70710678, - 0.70710678,  0.]])
cvtMatrices['lab2lms'] = np.linalg.inv(cvtMatrices['lms2lab'])


def Normalize(image):
    return image.astype('float32') / np.float32(255)


def Unnormalize(image):
    return image.astype('float32') * np.float32(255)


def gammaCorrection(image, gamma=1.0):
    return np.power(image, 1.0 / gamma)


def convertRGB2LAB(image):
    image = convertColorSpace(image, 'rgb2lms')
    image = convertColorSpace(image, 'lms2lab')
    return image


def convertLAB2RGB(image):
    image = convertColorSpace(image, 'lab2lms')
    image = convertColorSpace(image, 'lms2rgb')
    return image


def convertColorSpace(image, flag):
    imageTmp = image.copy()

    if flag == 'lms2lab':
        # avoid to log0
        imageTmp[imageTmp == 0.0] = np.float32(0.00000000001)
        imageTmp = np.log10(imageTmp).dot(cvtMatrices[flag].T)
    elif flag == 'lab2lms':
        imageTmp = np.power(10, imageTmp.dot(cvtMatrices[flag].T))
    else:
        imageTmp = imageTmp.dot(cvtMatrices[flag].T)
    return imageTmp


def convertGrayWorld(channel):
    return channel - np.median(channel)


def splitChannels(image):
    return np.dsplit(image, 3)


def stackChannels(channels):
    return np.dstack(channels)


def clipChannel(channel):
    channel = np.clip(channel, 0, 255)
    return channel


def correctColor(image):
    gamma = 0.2

    image = Normalize(image)
    image = gammaCorrection(image, gamma=gamma)
    image = convertRGB2LAB(image)

    l, a, b = splitChannels(image)
    a, b = convertGrayWorld(a), convertGrayWorld(b)
    image = stackChannels((l, a, b))

    image = convertLAB2RGB(image)
    image = gammaCorrection(image, gamma=1/gamma)
    image = Unnormalize(image)

    # fit 0 ~ 255
    r, g, b = splitChannels(image)
    r, g, b = clipChannel(r), clipChannel(g), clipChannel(b)
    image = stackChannels((r, g, b))

    return image


def main(imgPath):
    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    showImage(np.uint8(image))
    image = correctColor(image)
    showImage(np.uint8(image))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite('./output/' + os.path.basename(imgPath), image)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please give a file as argument.')
    elif not os.path.isfile(sys.argv[1]):
        print(sys.argv[1] + " : Not a file or file doesn't exist")
    else:
        main(sys.argv[1])
