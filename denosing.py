# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:43:42 2020

@author: Ivana Shahrear and Nasrin Akter
"""

############################ Denoising filters ###############
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import cv2




s = r'F:\\thesis\\Automatic renal cysts detection and segementation\\images'
image_no = '\im3.jpg'
s = s + image_no

img = cv2.imread(s,0)
plt.imshow(img, 'gray')

#img = img_as_float(io.imread("F:\\thesis\\Automatic renal cysts detection and segementation\\images\\image6.jpg"))
#Need to convert to float as we will be doing math on the array

from scipy import ndimage as nd


############  Non local mean filtering##########################

def non_loacl_filter(img):
    
    img = img_as_float(img)
    #Need to convert to float as we will be doing math on the array
    sigma_est = np.mean(estimate_sigma(img, multichannel=True))

    patch_kw = dict(patch_size=2,      
                patch_distance=7,  
                multichannel=True)

    denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                               patch_size=2, patch_distance=7, multichannel=True)
    """
    denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
    """
    imge = img_as_ubyte(denoise_img)
    plt.figure(figsize=(60,60))
    plt.subplot(121)
    plt.title("original image")
    plt.imshow(img,'gray')
    plt.subplot(122)

    plt.imshow(denoise_img, 'gray')
    plt.title(" denosing image")
    #plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')
    #plt.imsave("denosing image/NLM.jpg",denoise_img)
    return imge
##################### other filtering####################################
def build_filters():
    #returns a list of kernels in several orientations
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize': (ksize, ksize), 'sigma': 0.0225, 'theta': theta, 'lambd': 15.0,
                  'gamma': 0.01, 'psi': 0, 'ktype': cv2.CV_32F}
        
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern, params))
    return filters


def process(img, filters):
    #returns the img filtered by the filter list
    accum = np.zeros_like(img)
    for kern, params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum
def Histeq(img):
    equ = cv2.equalizeHist(img)
    return equ

def GaborFilter(img):
    filters = build_filters()
    p = process(img, filters)
    return p
img1=non_loacl_filter(img)
plt.imshow(img1,cmap='gray')
img2 = GaborFilter(img1)
plt.imshow(img2,cmap='gray')
img3 = Histeq(img2)
plt.imshow(img3,cmap='gray')
plt.subplot(121)
plt.title("original image",)
plt.imshow(img,'gray')
plt.subplot(122)

plt.imshow(img3, 'gray')
#plt.title(" denosing image")
#img3=cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

#plt.imsave("denosing image/filter.jpg",img3


