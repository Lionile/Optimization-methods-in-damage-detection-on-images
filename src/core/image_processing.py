import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN, KMeans
from skimage.metrics import structural_similarity as ssim
from skimage import color
import pywt # wavelets


def lowpass_filter2d(shape, threshold):
    pass



# doesnt work
def wavelet_denoise(img, wavelet='db1', level=1):
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruct
    denoised = pywt.waverec2(coeffs_H, wavelet)
    denoised = np.clip(denoised, 0, 255)

    return denoised[:img.shape[0], :img.shape[1]]





def threshold(img, threshold_min, threshold_max):
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
    img = img.copy()
    mask = np.zeros(img.shape)
    for index, pixel in np.ndenumerate(img):
        i, j = index
        if (img[i,j] >= threshold_min) and (img[i,j] <= threshold_max):
            mask[i,j] = 1
    return mask





def kmeans(img, k, min_group, max_group):
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





def dbscan(img, eps, threshold_min, threshold_max, min_samples = 5):
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