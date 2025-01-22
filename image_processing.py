import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN, KMeans
from skimage.metrics import structural_similarity as ssim
import pywt # wavelets


def lowpass_filter2d(shape, threshold):
    pass




def wavelet_filter(img, threshold, wavelet='haar', level=1):
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    cA = coeffs[0]
    cHVD = coeffs[1:]

    # default threshold
    # threshold = np.median(np.abs(cHVD[0][0])) / 0.6745

    # threshold
    for i in range(len(cHVD)):
        cH, cV, cD = cHVD[i]
        cHVD[i] = (pywt.threshold(cH, threshold, mode='soft'),
                   pywt.threshold(cV, threshold, mode='soft'),
                   pywt.threshold(cD, threshold, mode='soft'))

    # Reconstruct the image using inverse DWT
    filtered_img = pywt.waverec2([cA] + cHVD, wavelet)
    return filtered_img




'''
input:
    img - grayscale image
    threshold_min - minimum threshold value
    threshold_max - maximum threshold value

return:
    mask: binary mask for thresholded areas

optimizable params:
    0 <= threshold_min <= 255
    threshold_min <= threshold_max <=255
'''
def threshold(img, threshold_min, threshold_max):
    img = img.copy()
    mask = np.zeros(img.shape)
    for index, pixel in np.ndenumerate(img):
        i, j = index
        if (img[i,j] >= threshold_min) and (img[i,j] <= threshold_max):
            mask[i,j] = 1
    return mask




'''
input:
    img - 3d image array
    k - number of clusters in kmeans
    min_group - select n-th or greater group by size
    max_group - select up to n-th group by size
    together min_group and max_group form a bandpass

return:
    img: original image with thresholded areas marked in red
    mask: binary mask for thresholded areas

optimizable params:
    k > 1
    0 <= min_group < k-1
    min_group < max_group <= k
'''
def kmeans(img, k, min_group, max_group):
    img = img.copy()

    # random state for consistency
    # TODO: remove random state for evaluation
    kmeans = KMeans(k, random_state=0, n_init='auto').fit(img.reshape((-1, 3)))
    
    # sort groups by size
    label_dict = {}
    for i, label in enumerate(kmeans.labels_):
        if int(label) not in label_dict.keys():
            label_dict[int(label)] = 1
        else:
            label_dict[int(label)] += 1

    keys = [key for key in label_dict.keys()]
    label_counts = [label_dict[key] for key in keys]
    sorted_indices = np.argsort(label_counts)
    
    sorted_keys = [keys[i] for i in sorted_indices]

    # select specified clusters
    groups_to_remove = sorted_keys[min_group: max_group]
    
    predicted = kmeans.predict(img.reshape(-1, 3)).reshape(img.shape[:-1])
    mask = np.zeros(img.shape[:-1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if predicted[i,j] in groups_to_remove:
                mask[i,j] = 1
    for index, m in np.ndenumerate(mask):
        i, j = index
        if mask[i,j] == 1:
            img[i,j] = [255,0,0] # if masked, set to red
    return img, mask




'''
input:
    img - 3d image array
    eps - max distance to be part of a group
    min_samples - minimum number of points in a group
    threshold - percentage of smallest groups to take

return:
    img: original image with thresholded areas marked in red
    mask: binary mask for thresholded areas

optimizable params:
    eps > 0
    min_samples > 0
    threshold_min = [0, 1>
    threshold_max = <threshold_min, 1]
'''
def dbscan(img, eps, threshold_min, threshold_max, min_samples = 5):
    img = img.copy()

    # random state for consistency
    # TODO: remove random state for evaluation
    dbscan = DBSCAN(eps, min_samples=min_samples)
    predicted = dbscan.fit_predict(img.reshape((-1, 3))).reshape(img.shape[:-1])
    
    # sort groups by size
    label_dict = {}
    for i, label in enumerate(dbscan.labels_):
        if int(label) not in label_dict.keys():
            label_dict[int(label)] = 1
        else:
            label_dict[int(label)] += 1

    keys = [key for key in label_dict.keys()]
    label_counts = [label_dict[key] for key in keys]
    sorted_indices = np.argsort(label_counts)
    
    sorted_keys = [keys[i] for i in sorted_indices]

    # select specified clusters
    groups_to_remove = sorted_keys[int(threshold_min*len(sorted_keys)): int(threshold_max*len(sorted_keys))]
    
    mask = np.zeros(img.shape[:-1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if predicted[i,j] in groups_to_remove:
                mask[i,j] = 1
    for index, m in np.ndenumerate(mask):
        i, j = index
        if mask[i,j] == 1:
            img[i,j] = [255,0,0] # if masked, set to red
    return img, mask



'''
input:
    img1, img2 - 
'''
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr




def calculate_cnr(img1, img2):
    pass




'''
imput:
    img1, img2 - rgb or grayscale image
'''
def calculate_ssim(img1, img2):
    # convert to grayscale if they are in color
    if img1.ndim == 3:
        img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    
    ssim_value, _ = ssim(img1, img2, full=True)
    return ssim_value