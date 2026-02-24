
import cv2 as cv
from src.evaluation.loss_functions import *


def objective_gaussian(x, img1, img2, loss_function, loss_inverse=False, callback=None):
    '''
    input:
        x[0] - kernel_size [0, inf], Int
        x[1] - blur strength [0, inf] Float
        loss_function - on which loss function to evaluate
        loss_inverse - the algorithm calculates the minimum, if the point is to
            maximize the loss function, invert the value
        callback - callback for particle loss results at each iteration, used for particle history

    return:
        loss_score
    '''
    values = []
    for particle in x:
        kernel_size = int(particle[0])
        # make kernel size an odd integer
        if kernel_size % 2 == 0:
            kernel_size += 1
        modified = cv.GaussianBlur(img2, (kernel_size, kernel_size), particle[1])
        if loss_inverse:
            values.append(-loss_function(img1, modified))
        else:
            values.append(loss_function(img1, modified))
    if callback != None:
        callback(values)
    return values



def objective_median(x, img1, img2, loss_function, loss_inverse=False, callback=None):
    '''
    input:
        x[0] - kernel_size [0, inf], Int
        loss_function - on which loss function to evaluate
        loss_inverse - the algorithm calculates the minimum, if the point is to
            maximize the loss function, invert the value
        callback - callback for particle loss results at each iteration, used for particle history

    return:
        loss_score
    '''
    values = []
    for particle in x:
        kernel_size = int(particle[0])
        if kernel_size % 2 == 0:
            kernel_size += 1
        modified = cv.medianBlur(img2, kernel_size)
        if loss_inverse:
            values.append(-loss_function(img1, modified))
        else:
            values.append(loss_function(img1, modified))
    if callback != None:
        callback(values)
    return values



def objective_bilateral(x, img1, img2, loss_function, loss_inverse=False, callback=None):
    '''
    input:
        x[0] - diameter [0, inf], Int
        x[1] - sigmaColor [0, inf], Float
        x[2] - sigmaSpace [0, inf], Float
        loss_function - on which loss function to evaluate
        loss_inverse - the algorithm calculates the minimum, if the point is to
            maximize the loss function, invert the value
        callback - callback for particle loss results at each iteration, used for particle history

    return:
        loss_score
    '''
    values = []
    for particle in x:
        diameter = int(particle[0])
        sigma_color = particle[1]
        sigma_space = particle[2]
        modified = cv.bilateralFilter(img2, diameter, sigma_color, sigma_space)
        if loss_inverse:
            values.append(-loss_function(img1, modified))
        else:
            values.append(loss_function(img1, modified))
    if callback != None:
        callback(values)
    return values