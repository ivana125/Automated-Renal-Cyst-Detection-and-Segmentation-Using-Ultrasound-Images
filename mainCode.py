# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 21:16:35 2020

@author: Ivana Shahrear and Nasrin Akter
"""

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import scipy.ndimage
from skimage import morphology
from skimage import measure,color
from skimage.transform import resize
from sklearn.cluster import KMeans
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float


s = r'F:\\thesis\\Automatic renal cysts detection and segementation\\images'
image_no = '\im2.jpg'
s = s + image_no

img = cv2.imread(s,0)
width =476 
height = 378
dim = (width, height) 
im2=cv2.resize(img,dim, interpolation = cv2.INTER_AREA)


plt.imshow(img, 'gray')

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
########detection####################

def detection_cyst(img):
    
    params = cv2.SimpleBlobDetector_Params()

    # Define thresholds
    #Can define thresholdStep. See documentation. 
    params.minThreshold = 0
    params.maxThreshold = 87

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 50000

    # Filter by Color (black=0)
    params.filterByColor = True  #Set true for cast_iron as we'll be detecting black regions
    params.blobColor = 0

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.2
    params.maxCircularity = 3

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.01
    params.maxConvexity = 4

    # Filter by InertiaRatio
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 4

    # Distance Between Blobs
    params.minDistBetweenBlobs = 0.1

    # Setup the detector with parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(img)

    print("Number of blobs detected are : ", len(keypoints))
    
    # Draw blobs
    img_with_blobs = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #plt.imshow(img_with_blobs,"keypoints")
    cv2.imshow("Keypoints", img_with_blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    #cv2.imwrite("particle_blobs.jpg", img_with_blobs)
    return img_with_blobs 
    
def make_kidneymask(image,img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the kidney
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground 
    #
    kmeans = KMeans(n_clusters=7).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the kidney.  
    

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))#sure backgroud

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 255

    #
    #   we do another large dilation
    #  in order to fill in and out the kidney mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    print(dilation.dtype)
    print(mask.dtype)
    dilation=np.int32(dilation)
    unknown = cv2.subtract(thresh_img,img)
    # mp=mask*img
    ret3, markers = cv2.connectedComponents(np.uint8(thresh_img))
    markers = markers+10

    # Now, mark the region of unknown with zero
    markers[unknown==0] = 0
    
    markers = cv2.watershed(image,markers)
    #The boundary region will be marked -1
    #Let us color boundaries in yellow. OpenCv assigns boundaries to -1 after watershed.
    image[markers == -1] = [0,0,255]  

    img2 = color.label2rgb(markers, bg_label=0)

    cv2.imshow('original image', image)
    cv2.imshow('Colored Grains', img2)
    cv2.waitKey(0)

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels ,cmap='gray')
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img
img2=non_loacl_filter(im2)
img3 = GaborFilter(img2)
img3 = Histeq(img3)
#plt.subplot(121)
#plt.title("original image",)
plt.imshow(img,'gray')
#plt.subplot(122)

plt.imshow(img3, 'gray')
#plt.title(" denosing image")
#img3=cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

#plt.imsave("denosing image/filter.jpg",img3)
detectImage=detection_cyst(img3)
make_kidneymask(detectImage,img3, display=True) 
#cv2.waitKey(0)
##cv2.destroyAllWindows()    
