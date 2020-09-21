"""
Date: February/01/2020
Code - SIFT feature extraction
author: OmarJuarez16
@email: omarandre97@gmail.com
  
  Objective: This code merges the left and right images from a ZED stereo camera to obtain a single image. 
  Requirements: OpenCV-contrib-python 3.4.x, later versions will not work. Do not use for commercial reasons.  
  
  References: 
    - https://pylessons.com/OpenCV-image-stiching-continue/
    - https://www.youtube.com/watch?v=ToldvnUtBh0
    - https://www.youtube.com/watch?v=mY5jp8KQM6g
"""


# Importing libraries
import numpy as np 
import cv2 as cv


def stitch(left_image, right_image):  # Stitching through OpenCV SIFT (Scale-Invariant Feature Transform)
    
    left_gray = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()  # Call Sift class
    kp1, des1 = sift.detectAndCompute(left_gray, None)  # Find key points in left image
    kp2, des2 = sift.detectAndCompute(right_gray, None)  # Find key points in right image 

    match = cv.BFMatcher()  # Call Matcher class
    matches = match.knnMatch(des1, des2, k=2)  # Match the key points with KNN

    good = []  # List for collecting matches
    for m, n in matches:  # Loop to determine through a threshold whether is or isn't a desired match. 
        if m.distance < 0.6*n.distance:
            good.append(m)
    
    # Next two lines are for visualizing the matches. 
    # draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
    # img_matching_features = cv.drawMatches(left_image, kp1, right_image, kp2, good, None, **draw_params)

    min_match_count = 10  # Threshold of minimum matches  
    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # Coordinates of the points in the original plane
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # Coordinates of the points in the target plane
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)  # Homography between both planes through RANSAC method and a threshold of 5
        h, w = left_gray.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        img2 = cv.polylines(right_gray, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print('Not enough matches are founded, increase the distance please.')

    dst = cv.warpPerspective(left_image, M, (right_image.shape[1] + left_image.shape[1], right_image.shape[0]))
    dst[0:right_image.shape[0], 0:right_image.shape[1]] = right_image
    return dst, img2


def read_images(name):  # Reads the left and right images from the stereo camera. 
    
    left_image = '' + name  # Add the name for the left image and the root name. 
    right_image = '' + name  # Add the name for the right image and the root name. 
    
    # Reading of images
    left_image = cv.imread(left_image)  
    right_image = cv.imread(right_image)
    
    return left_image, right_image



def main(): 
  directory = ""  # Directory + root name of the images. 
  right_image, left_image = read_images(directory)
  image, _ = stitch(left_image, right_image)
  
  # Display image 
  cv.imshow("Stitched image", trim(dst))
  cv.waitkey(0)
  cv.destroyAllWindows() 
  # Save image
  cv.imwrite("NAME-OF-IMAGE", trim(dst))
  
  
if __name__ == "__main__":
    main()
