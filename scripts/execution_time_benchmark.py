import time
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.loss_functions import *
from src.optimization.noise_removal import *

def get_test_data(resize=True):
    # test data
    img_filenames = [
        '0076_004_N6_03200_00320_3200_L',
        '0096_005_N6_01600_01000_3200_L',
        '0120_006_N6_01600_00400_3200_L'
    ]

    images = [rf'data\raw\SIDD_Small_sRGB_Only\Data\{filename}\GT_SRGB_010.PNG' for filename in img_filenames]
    noisy_images = [rf'data\raw\SIDD_Small_sRGB_Only\Data\{filename}\NOISY_SRGB_010.PNG' for filename in img_filenames]


    images = [cv.cvtColor(cv.imread(img), cv.COLOR_BGR2RGB) for img in images]
    noisy_images = [cv.cvtColor(cv.imread(img), cv.COLOR_BGR2RGB) for img in noisy_images]

    # resize
    if resize:
        images = [cv.resize(image, None, fy=0.25, fx=0.25, interpolation=cv.INTER_LINEAR) for image in images]
        noisy_images = [cv.resize(image, None, fy=0.25, fx=0.25, interpolation=cv.INTER_LINEAR) for image in noisy_images]

    return images, noisy_images, img_filenames

def run_simple_test():
    # test data
    images, noisy_images, _ = get_test_data(resize=False)

    print(f'Running test with {len(images)} images')

    # test PSNR exection time
    start = time.time()

    for i in range(len(images)):
        calculate_psnr(images[i], noisy_images[i])

    end = time.time()
    print(f'PSNR time: {end-start}')



    # test SSIM exection time
    start = time.time()

    for i in range(len(images)):
        calculate_ssim(images[i], noisy_images[i])

    end = time.time()
    print(f'SSIM time: {end-start}')


def run_comlete_test():
    images, noisy_images, filenames = get_test_data()
    execution_times = {
        'loss_type': [],
        'filter_type': [],
        'execution_time': []
    }

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    pso_options = {
        'n_particles': 50,
        'dimensions': 3,
        'options': options,
        'bounds': ([0, 0, 0], [20, 100, 100]),
        'ftol': 1e-6,
        'ftol_iter': 3
    }
    
    for i, row in enumerate(zip(images, noisy_images, filenames)):
        img, img_noisy, image_filename = row
        # =====PSNR=====
        print(f'initial noise: {-calculate_psnr(img, img_noisy)}')
        # gaussian
        results_filename = rf'results/pso/{image_filename}_gaussian_psnr'
        start = time.time()
        gaussian_denoising(img, img_noisy, results_filename, calculate_psnr, True, pso_options, plot=False)
        end = time.time()
        execution_times['loss_type'].append('PSNR')
        execution_times['filter_type'].append('Gaussian')
        execution_times['execution_time'].append(end - start)

        # median
        results_filename = rf'results/pso/{image_filename}_median_psnr'
        start = time.time()
        median_denoising(img, img_noisy, results_filename, calculate_psnr, True, pso_options, plot=False)
        end = time.time()
        execution_times['loss_type'].append('PSNR')
        execution_times['filter_type'].append('Median')
        execution_times['execution_time'].append(end - start)

        # bilateral
        results_filename = rf'results/pso/{image_filename}_bilateral_psnr'
        start = time.time()
        bilateral_denoising(img, img_noisy, results_filename, calculate_psnr, True, pso_options, plot=False)
        end = time.time()
        execution_times['loss_type'].append('PSNR')
        execution_times['filter_type'].append('Bilateral')
        execution_times['execution_time'].append(end - start)

        # =====SSIM=====
        print(f'initial noise: {-calculate_ssim(img, img_noisy)}')
        # gaussian
        results_filename = rf'results/pso/{image_filename}_gaussian_ssim'
        start = time.time()
        gaussian_denoising(img, img_noisy, results_filename, calculate_ssim, True, pso_options, plot=False)
        end = time.time()
        execution_times['loss_type'].append('SSIM')
        execution_times['filter_type'].append('Gaussian')
        execution_times['execution_time'].append(end - start)

        # median
        results_filename = rf'results/pso/{image_filename}_median_ssim'
        start = time.time()
        median_denoising(img, img_noisy, results_filename, calculate_ssim, True, pso_options, plot=False)
        end = time.time()
        execution_times['loss_type'].append('SSIM')
        execution_times['filter_type'].append('Median')
        execution_times['execution_time'].append(end - start)

        # bilateral
        results_filename = rf'results/pso/{image_filename}_bilateral_ssim'
        start = time.time()
        bilateral_denoising(img, img_noisy, results_filename, calculate_ssim, True, pso_options, plot=False)
        end = time.time()
        execution_times['loss_type'].append('SSIM')
        execution_times['filter_type'].append('Bilateral')
        execution_times['execution_time'].append(end - start)
    df = pd.DataFrame(execution_times)
    df.to_csv('results/execution_times/complete_test.csv', index=False)




if __name__ == '__main__':
    # run_simple_test()
    run_comlete_test()