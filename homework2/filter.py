#!/usr/bin/env python
# -*- coding:utf-8 -*-

from skimage import io
from PIL import Image
import skimage
import skimage.filters
import numpy as np
import matplotlib.pyplot as plt
import functools
import time
import os


def _paddingFilling(image, m=3, n=3):
    up, down = image[0], image[-1]
    for i in range(m // 2):
        image = np.vstack([up, image, down])
    left, right = image[:, [0]], image[:, [-1]]
    for i in range(n // 2):
        image = np.hstack([left, image, right])
    return image


def _imageSpliting(image, m=3, n=3):
    height, width = image.shape
    oldImage = _paddingFilling(image, m, n)
    oldImages = []
    for i in range(m):
        for j in range(n):
            oldImages.append(oldImage[i:i + height, j:j + width])
    oldImages = np.asarray(oldImages)
    return oldImages


def arithmeticMeanFilter(image, m=3, n=3):
    oldImages = _imageSpliting(image, m=m, n=n)
    newImage = np.mean(oldImages, axis=0)
    return newImage


def geometricMeanFilter(image, m=3, n=3):
    oldImages = _imageSpliting(np.log(image + 1e-6), m=m, n=n)
    newImage = np.exp(np.mean(oldImages, axis=0))
    return newImage


def harmonicMeanFilter(image, m=3, n=3):
    oldImages = _imageSpliting(1 / (image + 1e-6), m=m, n=n)
    newImage = (1 / np.mean(oldImages, axis=0))
    return newImage


def inverseHarmonicMeanFilter(image, m=3, n=3, Q=1):
    oldImages1 = _imageSpliting((image + 1e-6) ** (Q + 1), m=m, n=n)
    oldImages2 = _imageSpliting((image + 1e-6) ** Q, m=m, n=n)
    return np.sum(oldImages1, axis=0) / np.sum(oldImages2, axis=0)


def medianFilter(image, m=3, n=3):
    oldImages = _imageSpliting(image, m=m, n=n)
    newImage = np.median(oldImages, axis=0)
    return newImage


def maximumFilter(image, m=3, n=3):
    oldImages = _imageSpliting(image, m=m, n=n)
    newImage = np.max(oldImages, axis=0)
    return newImage


def minimumFilter(image, m=3, n=3):
    oldImages = _imageSpliting(image, m=m, n=n)
    newImage = np.min(oldImages, axis=0)
    return newImage


def medianRangeFilterFilter(image, m=3, n=3):
    oldImages = _imageSpliting(image, m=m, n=n)
    newImage = (np.max(oldImages, axis=0) + np.min(oldImages, axis=0)) / 2
    return newImage


def improvedAlphaMeanFilter(image, m=3, n=3, d=2):
    d = d // 2
    oldImages = _imageSpliting(image, m=m, n=n)
    oldImages = np.sort(oldImages, axis=0)
    newImage = np.mean(oldImages[d:m * n - d], axis=0)
    return newImage


def AdaptiveMedianFilter(image, SMax=7):
    height, width = image.shape
    newImage = image.copy()
    for i in range(height):
        for j in range(width):
            filterSize = 3
            z = image[i][j]
            while(filterSize <= SMax):
                S = filterSize//2
                tmp = image[max(0, i-S): i+S+1, max(0, j-S): j+S+1].reshape(-1)
                tmp.sort()
                zMin = tmp[0]
                zMax = tmp[-1]
                zMed = tmp[len(tmp)//2]
                if(zMin < zMed and zMed < zMax):
                    if(z == zMin or z == zMax):
                        newImage[i][j] = zMed
                    break
                else:
                    filterSize += 2
    return newImage


def skimageGaussian(image, sigma=1):
    newImage = skimage.filters.gaussian(image, sigma=sigma)
    return newImage


def skimageMedian(image):
    newImage = skimage.filters.median(image)
    return newImage


def main():
    basePath = 'img'
    imagePath = os.path.join(basePath, 'lena512.bmp')
    image = io.imread(imagePath)
    mode = [
        'gaussian',
        'localvar',
        'poisson',
        'salt',
        'pepper',
        's&p',
        'speckle',
    ]
    filters = [
        arithmeticMeanFilter,
        geometricMeanFilter,
        harmonicMeanFilter,
        inverseHarmonicMeanFilter,
        medianFilter,
        maximumFilter,
        minimumFilter,
        medianRangeFilterFilter,
        improvedAlphaMeanFilter,
        AdaptiveMedianFilter,
        skimageGaussian,
        skimageMedian,
    ]
    for m in mode:
        print(m)
        path = os.path.join(basePath, m)
        if not os.path.exists(path):
            os.mkdir(path)
        imageNoise = skimage.util.random_noise(image, mode=m)
        savePath = os.path.join(path, '{}.png'.format(m))
        io.imsave(savePath, imageNoise)
        for f in filters:
            savePath = os.path.join(path, '{}_{}.png'.format(m, f.__name__))
            io.imsave(savePath, f(imageNoise))


if __name__ == "__main__":
    main()
