import time
import math
import numpy as np
import scipy as sp
import scipy.fft
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import cv2 as cv
from pyswarm import pso # minimizes the objective
import pyswarms as ps
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history
from scipy.optimize import differential_evolution
import pandas as pd
from noise import *
from image_processing import *
from loss_functions import *


image_filename = '0001_001_S6_00100_00060_3200_L'

# IMAGE LOADING AND PRE PROCESSING
img = cv.imread(rf'data\SIDD_Small_sRGB_Only\Data\{image_filename}\GT_SRGB_010.PNG')
img_noisy = cv.imread(rf'data\SIDD_Small_sRGB_Only\Data\{image_filename}\NOISY_SRGB_010.PNG')

# make the images smaller
img = cv.resize(img, None, fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
img_noisy = cv.resize(img_noisy, None, fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)





img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_noisy = cv.cvtColor(img_noisy, cv.COLOR_BGR2RGB)
# maximize gaussian parameters
# img_noisy, noise = add_gaussian_noise(img, 1)
# img_noisy, noise = add_salt_pepper_noise(img, 0.05)
print(f'initial noise: {-calculate_ssim(img, img_noisy)}')
# pso
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
# GAUSSIAN DENOISING
# optimizer = GlobalBestPSO(n_particles=20, dimensions=2, options=options, bounds=([1, 0], [9, 10]), ftol=-np.inf)
# cost, pos = optimizer.optimize(objective_gaussian, iters=100, img1=img, img2=img_noisy)

# kernel_size = int(pos[0])
# if kernel_size % 2 == 0:
#     kernel_size += 1
# print(f'Best result [gaussian]: strength: {pos[1]}, kernel: ({kernel_size},{kernel_size}), {cost}')

# _, ax = plt.subplots(1, 3)
# ax[0].imshow(img)
# ax[1].imshow(img_noisy)
# modified = cv.GaussianBlur(img_noisy, (kernel_size,kernel_size), pos[1])
# ax[2].imshow(modified)
# plt.show()



# MEDIAN DENOISING
# optimizer = GlobalBestPSO(n_particles=20, dimensions=1, options=options, bounds=([1], [9]), ftol=-np.inf)
# cost, pos = optimizer.optimize(objective_median, iters=100, img1=img, img2=img_noisy)

# kernel_size = int(pos[0])
# if kernel_size % 2 == 0:
#     kernel_size += 1
# print(f'Best result [median]: kernel: ({kernel_size},{kernel_size}), {cost}')

# _, ax = plt.subplots(1, 3)
# ax[0].imshow(img)
# ax[1].imshow(img_noisy)
# modified = cv.medianBlur(img_noisy, kernel_size)
# ax[2].imshow(modified)
# plt.show()



# BILATERAL DENOISING
results_filename = rf'results/pso/bilateral_{image_filename}'
optimizer = GlobalBestPSO(n_particles=50, dimensions=3, options=options, bounds=([0, 0, 0], [20, 100, 100]), ftol=1e-6, ftol_iter=3)
cost, pos = optimizer.optimize(objective_bilateral, iters=100, img1=img, img2=img_noisy)

print(f'Best result [bilateral]: d:{int(pos[0])}, sColor: {pos[1]}, sSpace: {pos[2]}, {cost}')

# find the best end particle
best_particle = 0
best_loss = objective_bilateral([optimizer.pos_history[-1][0]], img, img_noisy)
for i, particle in enumerate(optimizer.pos_history[-1]):
    loss = objective_bilateral([particle], img, img_noisy)
    if loss < best_loss:
        best_loss = loss
        best_particle = i

pos_history = np.array(optimizer.pos_history)[:,best_particle]

# save data with parameters from the best particle
data = {
    'diameter': pos_history[:,0],
    'sigmaColor': pos_history[:,1],
    'sigmaSpace': pos_history[:,2],
    'cost': optimizer.cost_history
}
df = pd.DataFrame(data)
df.to_csv(rf'{results_filename}/results.csv', index=False)
modified = cv.bilateralFilter(img_noisy, int(pos[0]), pos[1], pos[2])
cv.imwrite(rf'{results_filename}/filtered_image.png', modified)


_, ax = plt.subplots(1, 3)
ax[0].imshow(img)
ax[1].imshow(img_noisy)
ax[2].imshow(modified)
plt.show()