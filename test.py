import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from loss_functions import *


filename = rf'data\SIDD_Small_sRGB_Only\Data\0076_004_N6_03200_00320_3200_L'
img = cv.imread(rf'{filename}\GT_SRGB_010.PNG')
noisy_img = cv.imread(rf'{filename}\NOISY_SRGB_010.PNG')

print(calculate_psnr(img, noisy_img))