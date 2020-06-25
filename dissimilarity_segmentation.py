
# coding: utf-8

# In[1]:


import cv2
import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
import imutils
from diffimg import diff  #you may install it using 'pip install diffimg'
from PIL import Image
from pystackreg import StackReg


# In[2]:


def drawMatches(img1, kp1, img2, kp2, matches):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1] = np.dstack([img1])
    out[:rows2,cols1:] = np.dstack([img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0, 1), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0, 1), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0, 1), 1)

    return out


# In[3]:


def compare(filename1, filename2):
    img1 = cv2.imread(filename1)          # queryImage
    img2 = cv2.imread(filename2)          # trainImage
    img1 = cv2.detailEnhance(img1, sigma_s=10, sigma_r=0.15)  #it is not always necessary to use detail enhance. please check results in both cases and use best results.
    img2 = cv2.detailEnhance(img2, sigma_s=10, sigma_r=0.15)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()   #sift is patented and not free in latest version so you may try older versions.

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)

    matches = sorted(matches, key=lambda val: val.distance)

    img3 = drawMatches(img1,kp1,img2,kp2,matches[:40])

    # Show the image
    #cv2.imshow('Matched Features', img3)
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')
    return kp1,kp2,des1,des2,matches,img3


# In[4]:


def align(im1, im2):                                           #im1 and im2 are address of each image
    kp1,kp2,des1,des2,matches,img=compare(im1, im2)
    image_to_compare = cv2.imread(im1)          # queryImage
    original = cv2.imread(im2)
    image_to_compare = cv2.detailEnhance(image_to_compare, sigma_s=10, sigma_r=0.15)
    original = cv2.detailEnhance(original, sigma_s=10, sigma_r=0.15)
    result = cv2.drawMatches(image_to_compare, kp1, original, kp2, matches[0:200], None)
    #cv2.imwrite('pdftoimage/result/test_1_result.jpg', result)
    #cv2.imwrite('pdftoimage/result/test_1_img.jpg', img)
    
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, channels = original.shape
    im1Reg = cv2.warpPerspective(image_to_compare, h, (width, height))
    
    grayA = cv2.cvtColor(im1Reg, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    sr = StackReg(StackReg.AFFINE)
    out_aff = sr.register_transform(grayA, grayB) 
    
    #cv2.imwrite('im1.jpg', grayA)
    #cv2.imwrite('im2.jpg', out_aff)
    
    return grayA, out_aff
    


# In[5]:


def SSIM_method(image1, image2):
    (score, diff) = compare_ssim(image1, image2, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return score, diff, thresh


# In[6]:


def diff_img_method(image1, image2):
    grayA = Image.fromarray(image1)
    out_aff = Image.fromarray(image2)
    cv2.imwrite('im1.jpg', grayA)
    cv2.imwrite('im2.jpg', out_aff)
    diff('im1.jpg', 
     'im2.jpg', 
     delete_diff_file=False, 
     diff_img_file='result_diffimg/diff_img.jpg',
     ignore_alpha=False)
    good_diff = cv2.imread('result_diffimg/diff_img.jpg')
    enhanced_diff = cv2.detailEnhance(good_diff, sigma_s=50, sigma_r=0.15)
    #cv2.imwrite('result_diffimg/enhanced_diff.jpg', enhanced_diff)
    good_diff = cv2.cvtColor(good_diff, cv2.COLOR_BGR2GRAY)
    enhanced_diff = cv2.cvtColor(enhanced_diff, cv2.COLOR_BGR2GRAY)
    
    thresh1 = cv2.threshold(good_diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    thresh2 = cv2.threshold(enhanced_diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    #cv2.imwrite('result_diffimg/thresh1.jpg', thresh1)
    #cv2.imwrite('result_diffimg/thresh2.jpg', thresh2)
    
    return good_diff, enhanced_diff, thresh1, thresh2


# In[ ]:


image1, image2=align('pdftoimage/test_1/test_1-1.jpg', 'pdftoimage/test_2/test_2-1.jpg')


# In[ ]:


score_ssim, diff_ssim, thresh_ssim = SSIM_method(image1, image2)


# In[ ]:


diff2, enhanced_diff2, thresh1, thresh2 = diff_img_method(image1, image2)

