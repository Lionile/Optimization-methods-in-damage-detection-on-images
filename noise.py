import cv2 as cv
import numpy as np
from skimage.util import random_noise

def add_gaussian_noise(img, strength=0.1):
    new_img = img.copy()
    noise = np.random.normal(0, strength, new_img.shape).astype(np.float32)
    new_img = new_img + noise
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    return new_img, noise




def add_salt_pepper_noise(img, strength):
    new_img = img.copy()
    noise = random_noise(new_img, mode='s&p', amount=strength)
    new_img = np.clip(noise * 255, 0, 255).astype(np.uint8)
    return new_img, noise




def add_poisson_noise(img):
    new_img = img.copy()
    new_img = np.random.poisson(new_img).astype(np.uint8)
    return new_img




def add_speckle_noise(img):
    new_img = img.copy()
    noise = random_noise(img, mode='speckle', mean=0, var=0.1)
    new_img = np.clip(noise * 255, 0, 255).astype(np.uint8)
    return new_img