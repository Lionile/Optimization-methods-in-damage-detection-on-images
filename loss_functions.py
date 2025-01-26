import cv2 as cv
from noise import *
from image_processing import *




def calculate_psnr(img1, img2):
    '''
    input:
        img1, img2 - rgb or grayscale image
    '''
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr




def calculate_cnr(img1, img2):
    pass





def calculate_ssim(img1, img2):
    '''
    imput:
        img1, img2 - rgb or grayscale image
    '''
    # convert to grayscale if they are in color
    if img1.ndim == 3:
        img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    
    ssim_value, _ = ssim(img1, img2, full=True)
    return ssim_value




def calculate_psnr_edge(img1, img2):
    '''
    imput:
        img1, img2 - rgb or grayscale image
    '''
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))


    # convert to grayscale if they are in color
    if img1.ndim == 3:
        img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    
    # detect edges (horizontal, vertical)
    # img 1
    sobel_x1 = cv.Sobel(img1, cv.CV_64F, 1, 0, ksize=3)
    sobel_y1 = cv.Sobel(img1, cv.CV_64F, 0, 1, ksize=3)
    edges1 = np.hypot(sobel_x1, sobel_y1)
    # img 2
    sobel_x2 = cv.Sobel(img2, cv.CV_64F, 1, 0, ksize=3)
    sobel_y2 = cv.Sobel(img2, cv.CV_64F, 0, 1, ksize=3)
    edges2 = np.hypot(sobel_x2, sobel_y2)

    edges1 = cv.normalize(edges1, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    edges2 = cv.normalize(edges2, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # ssim on edges
    ssim_edge_value = calculate_ssim(edges1, edges2)

    # assign each ssim a weight
    combined_ssim = 1/40 * psnr + 1.5 * ssim_edge_value

    return combined_ssim




def calculate_ssim_edge(img1, img2):
    '''
    imput:
        img1, img2 - rgb or grayscale image
    '''
    # convert to grayscale if they are in color
    if img1.ndim == 3:
        img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    
    # detect edges (horizontal, vertical)
    # img 1
    sobel_x1 = cv.Sobel(img1, cv.CV_64F, 1, 0, ksize=3)
    sobel_y1 = cv.Sobel(img1, cv.CV_64F, 0, 1, ksize=3)
    edges1 = np.hypot(sobel_x1, sobel_y1)
    # img 2
    sobel_x2 = cv.Sobel(img2, cv.CV_64F, 1, 0, ksize=3)
    sobel_y2 = cv.Sobel(img2, cv.CV_64F, 0, 1, ksize=3)
    edges2 = np.hypot(sobel_x2, sobel_y2)

    edges1 = cv.normalize(edges1, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    edges2 = cv.normalize(edges2, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # original ssim
    ssim_value = calculate_ssim(img1, img2)

    # ssim on edges
    ssim_edge_value = calculate_ssim(edges1, edges2)

    # assign each ssim a weight
    combined_ssim = 0.5 * ssim_value + 0.5 * ssim_edge_value

    return combined_ssim