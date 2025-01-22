import cv2 as cv
from noise import *
from image_processing import *

'''
input:
    x[0] - kernel_size [0, inf], Int
    x[1] - blur strength [0, inf] Float

return:
    loss_score
'''
def objective_gaussian(x, img1, img2):
    values = []
    for particle in x:
        kernel_size = int(particle[0])
        # make kernel size an odd integer
        if kernel_size % 2 == 0:
            kernel_size += 1
        modified = cv.GaussianBlur(img2, (kernel_size, kernel_size), particle[1])
        values.append(calculate_psnr(img1, modified))
    return values


'''
input:
    x[0] - kernel_size [0, inf], Int

return:
    loss_score
'''
def objective_median(x, img1, img2):
    values = []
    for particle in x:
        kernel_size = int(particle[0])
        if kernel_size % 2 == 0:
            kernel_size += 1
        modified = cv.medianBlur(img2, kernel_size)
        values.append(calculate_psnr(img1, modified))
    return values


'''
input:
    x[0] - diameter [0, inf], Int
    x[1] - sigmaColor [0, inf], Float
    x[2] - sigmaSpace [0, inf], Float

return:
    loss_score
'''
def objective_bilateral(x, img1, img2):
    values = []
    for particle in x:
        diameter = int(particle[0])
        sigma_color = particle[1]
        sigma_space = particle[2]
        modified = cv.bilateralFilter(img2, diameter, sigma_color, sigma_space)
        values.append(-calculate_ssim(img1, modified))
    return values