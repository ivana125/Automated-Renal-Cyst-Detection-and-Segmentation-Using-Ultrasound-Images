# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:18:35 2020

@author: Ivana Shahrear and Nasrin Akter
"""

import cv2
%matplotlib inline

import numpy as np
#import dicom
import os
import matplotlib.pyplot as plt
from glob import glob
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure,color
from skimage.transform import resize
from sklearn.cluster import KMeans



 
img= cv2.imread("F:\\thesis\\Automatic renal cysts detection and segementation\\images\\im1.jpg")

width =476 
height = 378
dim = (width, height) 
im2=cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
img1=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
#print(originalImage.shape)


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
    markers[unknown==255] = 0
    
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
make_kidneymask(img,img1, display=True) 
#cv2.waitKey(0)
##cv2.destroyAllWindows()
