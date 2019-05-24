import numpy as np
import cv2
from matplotlib import pyplot as plt


def read_raw_image(url, rows, cols, color=False):
    if color:
        fd = open(url, 'rb')
        file = np.fromfile(fd, dtype=np.uint8, count=rows*cols*3)
        fd.close()
        return file.reshape((rows, cols, 3))
    else:
        fd = open(url, 'rb')
        file = np.fromfile(fd, dtype=np.uint8, count=rows*cols)
        fd.close()
        return file.reshape((rows, cols))


def show_images_with_plt(datas, cols):
    plt.figure(num=None, figsize=(
        18, 18 * (((len(datas) - 1)//cols) + 1) / cols), dpi=94)
    for index, data in enumerate(datas):
        plt.subplot((len(datas) - 1)//cols + 1, cols, index+1)
        plt.imshow(data["image"], cmap=plt.cm.gray, vmin=0, vmax=255)
        plt.axis("off")
        plt.title(data["title"])


def show_images_hist(datas, cols):
    plt.figure(num=None, figsize=(18, 4), dpi=94)
    for index, data in enumerate(datas):
        plt.subplot(len(datas)/cols+1, cols, index+1)
        plt.hist(data["image"].ravel(), bins=256, range=(0, 255))
        plt.title(data["title"])


def show_images_cdf(datas, cols):
    plt.figure(num=None, figsize=(18, 4), dpi=94)
    for index, data in enumerate(datas):
        hist, bin_edges = np.histogram(data["image"], bins=256, range=(0, 255))
        cdf = np.cumsum(hist/256/256)
        plt.subplot(len(datas)/cols+1, cols, index+1)
        plt.plot(cdf)
        plt.title(data["title"] + "\nSuggest threshold: " +
                  str(np.where(cdf > 0.70)[0][0]))
