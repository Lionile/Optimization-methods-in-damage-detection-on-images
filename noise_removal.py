import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# from pyswarm import pso
import pyswarms as ps
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history
from scipy.optimize import differential_evolution
import pandas as pd
from noise import *
from image_processing import *
from loss_functions import *
from objective_functions import *





# GAUSSIAN DENOISING
def gaussian_denoising(img, img_noisy, results_filename, loss_function, loss_inverse, pso_options, plot):
    if not os.path.exists(results_filename): os.makedirs(results_filename)

    fitness_values = []
    def callback(values):
        fitness_values.append(values)

    optimizer = GlobalBestPSO(**pso_options)
    cost, pos = optimizer.optimize(objective_gaussian, iters=100,
                                   img1=img, img2=img_noisy, loss_function=loss_function, loss_inverse=loss_inverse, callback=callback)

    kernel_size = int(pos[0])
    if kernel_size % 2 == 0:
        kernel_size += 1
    print(f'Best result [gaussian]: strength: {pos[1]}, kernel: ({kernel_size},{kernel_size}), {cost}')


    # find the best end particle
    best_particle = 0
    best_loss = objective_gaussian([optimizer.pos_history[-1][0]], img, img_noisy, loss_function=loss_function, loss_inverse=loss_inverse)
    for i, particle in enumerate(optimizer.pos_history[-1]):
        loss = objective_gaussian([particle], img, img_noisy, loss_function=loss_function, loss_inverse=loss_inverse)
        if loss < best_loss:
            best_loss = loss
            best_particle = i

    pos_history = np.array(optimizer.pos_history)[:,best_particle]
    average_loss_per_iteration = [float(np.average(x)) for x in fitness_values]

    # save data with parameters from the best particle
    data = {
        'kernel_size': pos_history[:,0],
        'sigma': pos_history[:,1],
        'average_cost': average_loss_per_iteration,
        'best_cost': optimizer.cost_history
    }
    df = pd.DataFrame(data)
    df.to_csv(rf'{results_filename}/results.csv', index=False)
    modified = cv.GaussianBlur(img_noisy, (kernel_size, kernel_size), pos[1])
    image_save = cv.cvtColor(modified, cv.COLOR_RGB2BGR)
    cv.imwrite(rf'{results_filename}/filtered_image.png', image_save)
    
    if plot:
        _, ax = plt.subplots(1, 3)
        ax[0].imshow(img)
        ax[1].imshow(img_noisy)
        modified = cv.GaussianBlur(img_noisy, (kernel_size,kernel_size), pos[1])
        ax[2].imshow(modified)
        plt.show()
    
    return modified



# MEDIAN DENOISING
def median_denoising(img, img_noisy, results_filename, loss_function, loss_inverse, pso_options, plot):
    if not os.path.exists(results_filename): os.makedirs(results_filename)

    fitness_values = []
    def callback(values):
        fitness_values.append(values)
    
    optimizer = GlobalBestPSO(**pso_options)
    cost, pos = optimizer.optimize(objective_median, iters=100,
                                   img1=img, img2=img_noisy, loss_function=loss_function, loss_inverse=loss_inverse, callback=callback)

    kernel_size = int(pos[0])
    if kernel_size % 2 == 0:
        kernel_size += 1
    print(f'Best result [median]: kernel: ({kernel_size},{kernel_size}), {cost}')

    # find the best end particle
    best_particle = 0
    best_loss = objective_median([optimizer.pos_history[-1][0]], img, img_noisy, loss_function=loss_function, loss_inverse=loss_inverse)
    for i, particle in enumerate(optimizer.pos_history[-1]):
        loss = objective_median([particle], img, img_noisy, loss_function=loss_function, loss_inverse=loss_inverse)
        if loss < best_loss:
            best_loss = loss
            best_particle = i

    pos_history = np.array(optimizer.pos_history)[:,best_particle]
    average_loss_per_iteration = [float(np.average(x)) for x in fitness_values]

    # save data with parameters from the best particle
    data = {
        'kernel_size': pos_history[:,0],
        'average_cost': average_loss_per_iteration,
        'best_cost': optimizer.cost_history
    }
    df = pd.DataFrame(data)
    df.to_csv(rf'{results_filename}/results.csv', index=False)
    modified = cv.medianBlur(img_noisy, kernel_size)
    image_save = cv.cvtColor(modified, cv.COLOR_RGB2BGR)
    cv.imwrite(rf'{results_filename}/filtered_image.png', image_save)

    if plot:
        _, ax = plt.subplots(1, 3)
        ax[0].imshow(img)
        ax[1].imshow(img_noisy)
        modified = cv.medianBlur(img_noisy, kernel_size)
        ax[2].imshow(modified)
        plt.show()
    
    return modified



# BILATERAL DENOISING
def bilateral_denoising(img, img_noisy, results_filename, loss_function, loss_inverse, pso_options, plot):
    if not os.path.exists(results_filename): os.makedirs(results_filename)

    fitness_values = []
    def callback(values):
        fitness_values.append(values)

    optimizer = GlobalBestPSO(**pso_options)
    cost, pos = optimizer.optimize(objective_bilateral, iters=100,
                                img1=img, img2=img_noisy, loss_function=loss_function, loss_inverse=loss_inverse, callback=callback)

    print(f'Best result [bilateral]: d:{int(pos[0])}, sColor: {pos[1]}, sSpace: {pos[2]}, {cost}')

    # find the best end particle
    best_particle = 0
    best_loss = objective_bilateral([optimizer.pos_history[-1][0]], img, img_noisy, loss_function=loss_function, loss_inverse=loss_inverse)
    for i, particle in enumerate(optimizer.pos_history[-1]):
        loss = objective_bilateral([particle], img, img_noisy, loss_function=loss_function, loss_inverse=loss_inverse)
        if loss < best_loss:
            best_loss = loss
            best_particle = i

    pos_history = np.array(optimizer.pos_history)[:,best_particle]
    average_loss_per_iteration = [float(np.average(x)) for x in fitness_values]

    # save data with parameters from the best particle
    data = {
        'diameter': pos_history[:,0],
        'sigmaColor': pos_history[:,1],
        'sigmaSpace': pos_history[:,2],
        'average_cost': average_loss_per_iteration,
        'best_cost': optimizer.cost_history
    }
    df = pd.DataFrame(data)
    df.to_csv(rf'{results_filename}/results.csv', index=False)
    modified = cv.bilateralFilter(img_noisy, int(pos[0]), pos[1], pos[2])
    image_save = cv.cvtColor(modified, cv.COLOR_RGB2BGR)
    cv.imwrite(rf'{results_filename}/filtered_image.png', image_save)

    if plot:
        _, ax = plt.subplots(1, 3)
        ax[0].imshow(img)
        ax[1].imshow(img_noisy)
        ax[2].imshow(modified)
        plt.tight_layout()
        plt.show()

    # return final image
    return modified



if __name__ == '__main__':
    # IMAGE LOADING AND PRE PROCESSING
    image_filename = '0001_001_S6_00100_00060_3200_L'       # 1
    # image_filename = '0036_002_GP_06400_03200_3200_N'       # 2
    # image_filename = '0059_003_G4_00800_01000_5500_L'       # 3
    # image_filename = '0076_004_N6_03200_00320_3200_L'       # 4
    # image_filename = '0096_005_N6_01600_01000_3200_L'       # 5
    # image_filename = '0120_006_N6_01600_00400_3200_L'       # 6

    img = cv.imread(rf'data\SIDD_Small_sRGB_Only\Data\{image_filename}\GT_SRGB_010.PNG')
    img_noisy = cv.imread(rf'data\SIDD_Small_sRGB_Only\Data\{image_filename}\NOISY_SRGB_010.PNG')

    # make the images smaller
    img = cv.resize(img, None, fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
    img_noisy = cv.resize(img_noisy, None, fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)

    # convert to rgb
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_noisy = cv.cvtColor(img_noisy, cv.COLOR_BGR2RGB)

    # add different/aditional noise
    # img_noisy, noise = add_gaussian_noise(img, 1)
    # img_noisy, noise = add_salt_pepper_noise(img, 0.05)





    # # ==========PSO==========
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    pso_options = {
        'n_particles': 50,
        'dimensions': 3,
        'options': options,
        'bounds': ([0, 0, 0], [20, 100, 100]),
        'ftol': 1e-6,
        'ftol_iter': 3
    }


    # =====PSNR=====
    # print(f'initial noise: {-calculate_psnr(img, img_noisy)}')
    # # gaussian
    # results_filename = rf'results/pso/{image_filename}_gaussian_psnr'
    # gaussian_denoising(img, img_noisy, results_filename, calculate_psnr, True, pso_options, plot=False)

    # # median
    # results_filename = rf'results/pso/{image_filename}_median_psnr'
    # median_denoising(img, img_noisy, results_filename, calculate_psnr, True, pso_options, plot=False)

    # # bilateral
    # results_filename = rf'results/pso/{image_filename}_bilateral_psnr'
    # bilateral_denoising(img, img_noisy, results_filename, calculate_psnr, True, pso_options, plot=False)


    # =====SSIM=====
    # print(f'initial noise: {-calculate_ssim(img, img_noisy)}')
    # # gaussian
    # results_filename = rf'results/pso/{image_filename}_gaussian_ssim'
    # gaussian_denoising(img, img_noisy, results_filename, calculate_ssim, True, pso_options, plot=False)

    # # median
    # results_filename = rf'results/pso/{image_filename}_median_ssim'
    # median_denoising(img, img_noisy, results_filename, calculate_ssim, True, pso_options, plot=False)

    # # bilateral
    # results_filename = rf'results/pso/{image_filename}_bilateral_ssim'
    # bilateral_denoising(img, img_noisy, results_filename, calculate_ssim, True, pso_options, plot=False)



    # =====PSNR + EDGE PRESERVATION=====
    print(f'initial noise: {-calculate_psnr_edge(img, img_noisy)}')
    # gaussian
    results_filename = rf'results/pso/{image_filename}_gaussian_psnr-edge'
    gaussian_denoising(img, img_noisy, results_filename, calculate_psnr_edge, True, pso_options, plot=False)

    # median
    # results_filename = rf'results/pso/{image_filename}_median_psnr-edge'
    # median_denoising(img, img_noisy, results_filename, calculate_psnr_edge, True, pso_options, plot=False)

    # bilateral
    # results_filename = rf'results/pso/{image_filename}_bilateral_psnr-edge'
    # bilateral_denoising(img, img_noisy, results_filename, calculate_psnr_edge, True, pso_options, plot=False)



    # =====SSIM + EDGE PRESERVATION=====
    # print(f'initial noise: {-calculate_ssim_edge(img, img_noisy)}')
    # # gaussian
    # results_filename = rf'results/pso/{image_filename}_gaussian_ssim-edge'
    # gaussian_denoising(img, img_noisy, results_filename, calculate_ssim_edge, True, pso_options, plot=False)

    # median
    # results_filename = rf'results/pso/{image_filename}_median_ssim-edge'
    # median_denoising(img, img_noisy, results_filename, calculate_ssim_edge, True, pso_options, plot=False)

    # bilateral
    # results_filename = rf'results/pso/{image_filename}_bilateral_ssim-edge'
    # bilateral_denoising(img, img_noisy, results_filename, calculate_ssim_edge, True, pso_options, plot=False)




    # plot HSV values of the image
    # _, ax = plt.subplots(1, 3, figsize=(12, 6))
    # img_hsv = cv.cvtColor(img_noisy, cv.COLOR_RGB2HSV)
    # for i in range(3):
    #     ax[i].imshow(img_hsv[:,:,i], cmap='gray')
    # plt.show()


    # plot YCrCb values of the image
    # _, ax = plt.subplots(2, 3, figsize=(12, 6))
    # img_ycbcr = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
    # img_noisy_ycbcr = cv.cvtColor(img_noisy, cv.COLOR_RGB2YCrCb)
    # for i in range(3):
    #     ax[0, i].imshow(img_ycbcr[:,:,i], cmap='gray')
    #     ax[1, i].imshow(img_noisy_ycbcr[:,:,i], cmap='gray')
    # plt.show()



    # wavelet denoise
    # _, ax = plt.subplots(1, 3, figsize=(12, 6))
    # img_ycbcr = cv.cvtColor(img_noisy, cv.COLOR_RGB2YCrCb)
    # img_ycbcr2 = img_ycbcr.copy()
    # img_ycbcr2[:,:,0] = wavelet_denoise(img_ycbcr[:,:,0])
    # denoised = cv.cvtColor(img_ycbcr, cv.COLOR_YCrCb2RGB)
    # ax[0].imshow(img)
    # ax[1].imshow(img_ycbcr[:,:,0])
    # ax[2].imshow(img_ycbcr2[:,:,0])
    # plt.show()
