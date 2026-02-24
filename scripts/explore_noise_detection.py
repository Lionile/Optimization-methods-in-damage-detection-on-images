import time
import math
import numpy as np
import scipy as sp
import scipy.fft
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import cv2 as cv
import pyswarms as ps
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history
from scipy.optimize import differential_evolution
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.noise import *
from src.core.image_processing import *
from src.evaluation.loss_functions import *




image_filename = '0001_001_S6_00100_00060_3200_L'

# IMAGE LOADING AND PRE PROCESSING
img = cv.imread(rf'data\raw\SIDD_Small_sRGB_Only\Data\{image_filename}\GT_SRGB_010.PNG')
img_noisy = cv.imread(rf'data\raw\SIDD_Small_sRGB_Only\Data\{image_filename}\NOISY_SRGB_010.PNG')

# make the images smaller
img = cv.resize(img, None, fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
img_noisy = cv.resize(img_noisy, None, fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)


#
# =========PRE PROCESSING=========
# plot HSV values of the image
# _, ax = plt.subplots(1, 3, figsize=(12, 6))
# for i in range(3):
#     ax[i].imshow(img_hsv[:,:,i], cmap='gray')
# plt.show()





# find defects in the image
# =========THRESHOLDING=========
# print(f'shape: {img_gray.shape}')
# mask = threshold(img_gray, 60, 75)
# mask = mask == 1
# img_rgb_threshold = cv.cvtColor(img_gray, cv.COLOR_GRAY2RGB)
# img_rgb_threshold[mask] = [255, 0, 0]

# plt.imshow(img_rgb_threshold)
# plt.show()




# =========KMEANS=========
# k = 20
# _, ax = plt.subplots(2, 3, figsize=(12, 6))
# for i in range(6):
#     min_group = i+6
#     max_group = i+6+1
#     kmeans_img, mask = kmeans(img_rgb, 20, min_group, max_group)
#     print(np.sum(mask))

#     ax[i//3, i%3].imshow(kmeans_img)

# plt.tight_layout()
# plt.show()




# =========DBSCAN=========
# eps = 1
# threshold_min = 0.7
# threshold_max = 0.82
# # img_rgb = cv.GaussianBlur(img_rgb, (3,3), 0)
# img_rgb_dbscan = cv.resize(img_rgb, (img_rgb.shape[1]//4, img_rgb.shape[0]//4)) # uses too much memory otherwise
# dbscan_img, mask = dbscan(img_rgb_dbscan, eps, threshold_min, threshold_max)
# plt.imshow(dbscan_img)
# plt.tight_layout()
# plt.show()
