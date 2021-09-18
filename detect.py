import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
%matplotlib inline
from skimage.color import rgb2gray
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float


img1 = cv2.imread("F:\\thesis\\Automatic renal cysts detection and segementation\\images\\im4.jpg")
width =476 
height = 378
dim = (width, height) 
im2=cv2.resize(img1,dim, interpolation = cv2.INTER_AREA)


plt.imshow(img1)
#imge=rgb2gray(img1)
###########################denoising#################
'''image=img_as_float(im2)

sigma_est = np.mean(estimate_sigma(image, multichannel=True))

patch_kw = dict(patch_size=5,      
                patch_distance=3,  
                multichannel=True)

imge = denoise_nl_means(im2, h=1.15 * sigma_est, fast_mode=False,
                               patch_size=5, patch_distance=3, multichannel=True)
"""
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)"""
""
imge = img_as_ubyte(imge)'''


img2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
plt.imshow(img2)
params = cv2.SimpleBlobDetector_Params()

# Define thresholds
#Can define thresholdStep. See documentation. 
params.minThreshold = 10
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
params.minDistBetweenBlobs = .1

# Setup the detector with parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(img2)

print("Number of blobs detected are : ", len(keypoints))

# Draw blobs
img_with_blobs = cv2.drawKeypoints(img2, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_with_blobs)
cv2.imshow("Keypoints", img_with_blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
#cv2.imwrite("particle_blobs.jpg", img_with_blobs)
