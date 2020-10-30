import rasterio
from datetime import datetime
from satpy.scene import Scene
from satpy import find_files_and_readers
from satpy.config import check_satpy
from satpy import available_readers
from pyresample import get_area_def
from skimage.morphology import disk, dilation
import numpy.ma as ma
from numpy import nan
from skimage import data, io, filters
from scipy.ndimage import gaussian_filter



import numpy as np 
import cv2
import matplotlib.pyplot as plt
from pprint import pprint
import math
from sklearn.metrics import mean_squared_error 
import skimage.color
import skimage.io
import statistics 
import time
from scipy.optimize import curve_fit
from scipy import ndimage, misc

from matplotlib.gridspec import GridSpec

from skimage import data, transform, exposure
from skimage.util import compare_images
from scipy import misc 
from scipy.io import savemat

start_time = time.time()

# Python code that does Image Registration using the RANSAC Algorithm


"*--------------CONSTANTS--------------*"    
K = 3 #  used for random samplin g in ransac_fit()
ITER_NUM = 2000 # Initial parameter used is 2000

# https://docs.opencv.org/3.4.9/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
max_runtime_count = 2

cols = 1000 #756 #756 #1000 #767 #692
rows =  1000 #596 #1000 #692 #767 #1211 width 767

upscaled_height_l8 = rows
upscaled_width_l8 = cols

NUM_OCTAVES = int(np.floor( math.log2(min(cols, rows))) - 3)
print("Number of Octaves:", NUM_OCTAVES)

RANSAC_THRESHOLD = 3   
SIGMA_BLUR = 1.6  
CONTRAST_THRESHOLD = 0.03 
EDGE_THRESHOLD = 10 

if (NUM_OCTAVES < 0):
    raise Exception('Check set area and upscaled height and width as Number of Octaves cannot be neagtive!')

"*-------------- Function Definitions --------------*"            
# Documentation on SIFT Features
def extract_SIFT_L8(img, Mask):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create() # Default SIFT parametg800ers (David Lowe Paper)
    sift= cv2.xfeatures2d.SIFT_create(nfeatures=0, # 0
                                nOctaveLayers=NUM_OCTAVES, # 3s
                                contrastThreshold=CONTRAST_THRESHOLD, # 0.03
                                edgeThreshold=EDGE_THRESHOLD, # 10.0
                                sigma=SIGMA_BLUR) # 1.6
    kp, desc = sift.detectAndCompute(img, Mask)
    # print("printing keypoints with SIFT function:", kp)
    # showing_sift_features(img, kp)
    x_y_coordinates_kp = np.array([p.pt for p in kp]).T 
    return x_y_coordinates_kp, kp, desc 

def extract_SIFT_S3(img, Mask):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create() # Default SIFT parameters (David Lowe Paper)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, # 0
                                nOctaveLayers=NUM_OCTAVES, # 3
                                contrastThreshold=0.03, # 0.03
                                edgeThreshold=10.0, # 10.0
                                sigma= 0.5) # 0.5
    # kp, desc = sift.detectAndCompute(img, Mask)
    # showing_sift_features(img, kp)
    # print("angle of first sift keypoint:", kp[0].angle)
    # print("Size of original keypoints:", len(kp))
    
    # Convert Keypoints to coordinates 
    x_y_coordinates_kp = np.array([p.pt for p in kp]).T 
    # print("Size of coordinate keypoints:", x_y_coordinates_kp.shape)
    return x_y_coordinates_kp, kp, desc 


def showing_sift_features(img1, key_points):
    plt.imshow(cv2.drawKeypoints(img1, key_points, None))
    plt.colorbar()
    plt.show() 

def match_SIFT(descriptor_source, descriptor_target):
    # bf = cv2.BFMatcher() 
    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)
    matches = bf.knnMatch(descriptor_source, descriptor_target, k=2)
    pos = np.array([], dtype=np.int32).reshape((0, 2))
    matches_num = len(matches)

    bf2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches2 = bf2.match(descriptor_source, descriptor_target)
    matches2 = sorted(matches2, key = lambda x:x.distance)

    # To uncomment numgoodmatches and matches_extra, GOOD_MATCH_PERCENT
    # numGoodMatches = int(len(matches) * 0.15)
    # matches_extra = matches[:numGoodMatches]
    
    for i in range(matches_num):
        if matches[i][0].distance <= 0.9 * matches[i][1].distance:
            temp = np.array([matches[i][0].queryIdx, 
                            matches[i][0].trainIdx])
            pos = np.vstack((pos, temp))
    return matches, pos, matches2

def estimate_affine(s, t):
    num = s.shape[1]
    M = np.zeros((2*num, 6))

    for i in range(num):
        temp = [[s[0, i], s[1,  i], 0, 0, 1, 0],
                [0, 0, s[0, i], s[1, i], 0, 1]]
        M[2 * i: 2 * i + 2, :] = np.array(temp)
    b = t.T.reshape((2 * num, 1))

    # Find the least-squares solution of M*x = b where theta = x 
    theta = np.linalg.lstsq(M, b, rcond=None)[0] # Shape of theta is (6,1)
    X = theta[:4].reshape((2, 2)) 
    Y = theta[4:]

    return X, Y 

def myestimate_affine(s, t):
    # Useful Link to understand this part:  http://ros-developer.com/2017/12/26/finding-affine-transform-with-linear-least-squares/#respond 
    num = s.shape[1]

    # print("Size of s:", s.shape)
    # print("Size of t:", t.shape)
    # print(num)
    A = np.zeros((2*num, 6))

    for i in range(num):
        temp = [[s[0, i], s[1,  i], 1, 0, 0, 0],
                [0, 0, 0, s[0, i], s[1, i], 1 ]]

        A[2 * i: 2 * i + 2, :] = np.array(temp)

    b = t.T.reshape((2 * num, 1))

    # print("Size of A:", A.shape)
    # print("Size of b:", b.shape)

    # Solving Ax = B where A contains the source coordinates and B contains the target/reference coordinates 
    # Method utilised is known as the Linear Least Squares Method 
    x = np.dot(np.linalg.pinv(A), b)
    # print("Size of x:", x.shape)

    # Reshape x array of size (6, 1) to a (3,3) matrix called X 
    # The homography matrix H = Transpose(X)
    X = np.zeros((3,3))

    X[0,0] = x[0]
    X[1,0] = x[1]
    X[2,0] = x[2]

    X[0,1] = x[3]
    X[1,1] = x[4]
    X[2,1] = x[5]

    X[0,2] = 0
    X[1,2] = 0
    X[2,2] = 1

    # https://www.mathworks.com/help/images/ref/affine2d.html
    # X has the following form:
    # tform = affine2d([ ...
    # cosd(theta) sind(theta) 0;...
    # -sind(theta) cosd(theta) 0; ...
    # atx aty 1])

    return X

# Function to fix gradient to 1
# def no_grad_line(vv, tx):
#     return(vv+tx)
def fit_func(a, b): # where a is v and b is tx and gradient is set to 1
    return a*1 + b

def residual_lengths(X, Y, s, t):
    e = np.dot(X, s) + Y
    diff_square = np.power(e - t, 2)
    residual = np.sqrt(np.sum(diff_square, axis=0))
    return residual

def ransac_fit(pts_s, pts_t):
    # Function taken from the book (Chapter 5) Image Processing Using Machine Learning by Singh 
    inliers_num =0 
    A = None
    t = None
    inliers = None

    # print("Largest size of random generation:", pts_s.shape[1])
    # print("Size of source points:",  pts_s.shape)

    for i in range(ITER_NUM):
        # np.random.seed(SEED) 
        idx = np.random.randint(0, pts_s.shape[1], (K, 1))
        A_tmp, t_tmp = estimate_affine(pts_s[:, idx], pts_t[:, idx])
        residual = residual_lengths(A_tmp, t_tmp, pts_s, pts_t)
        # print("Size of Residual length----------------------->:", residual.shape)
        if not(residual is None):
            inliers_tmp = np.where(residual < RANSAC_THRESHOLD )
            inliers_num_tmp = len(inliers_tmp[0])
            if inliers_num_tmp > inliers_num:
                inliers_num = inliers_num_tmp
                inliers = inliers_tmp
                A = A_tmp
                t = t_tmp
            else:
                pass
    return A, t, inliers

def myaffine_matrix(s, t, pos):
    s = s[:, pos[:, 0]]
    t = t[:, pos[:, 1]]
    # print("SIZE OF POINTS OF SOURCE in affine matrix fn:", s.shape[1])
    # print("SIZE OF POINTS OF TARGET in affine matrix fn:", t.shape[1])
    _, _, inliers = ransac_fit(s, t)
    # print("Number of inliers in affine matrix fn:", len(inliers[0]))
    # print("within Affine matrix:", inliers[0])
    s1 = s[:, inliers[0]]
    t1 = t[:, inliers[0]]
    # Finding X to solve Ax = B
    X = myestimate_affine(s1, t1)
    H = np.transpose(X)
    # inliers contains the number/indices of the chosen keypoints as the inliers 
    return H, inliers, s1, t1, s, t

def affine_matrix(s, t, pos):
    s = s[:, pos[:, 0]]
    t = t[:, pos[:, 1]]
    # print("SIZE OF POINTS OF SOURCE in affine matrix fn:", s.shape[1])
    # print("SIZE OF POINTS OF TARGET in affine matrix fn:", t.shape[1])
    _, _, inliers = ransac_fit(s, t)
    # print("Number of inliers in affine matrix fn:", len(inliers[0]))
    # print("within Affine matrix:", inliers[0])
    s = s[:, inliers[0]]
    t = t[:, inliers[0]]
    # inliers contains the number/indices of the chosen keypoints as the inliers 
    A, t1 = estimate_affine(s, t) # Compute homography from the inputted inliers
    M = np.hstack((A, t1))

    return M, inliers, s, t

# Capture_bits and cloud_confidence fns are utilised to extract the cloud mask from the L8 BQA TIFF file 
def _capture_bits(arr, b1, b2):
    width_int = int((b1 - b2 + 1) * "1", 2)
    return ((arr >> b2) & width_int).astype('uint8')

def cloud_confidence(arr):
    """
    00 = "Not Determined" = Algorithm did not determine the status of this condition
    01 = "No" = Algorithm has low to no confidence that this condition exists (0-33 percent confidence)
    10 = "Maybe" = Algorithm has medium confidence that this condition exists (34-66 percent confidence)
    11 = "Yes" = Algorithm has high confidence that this condition exists (67-100 percent confidence
    """
    return _capture_bits(arr, 6,5) #need to check these two integers are correct. See LS8 manual.

def _capture_bitsS3(arr, b1, b2):
    width_int = int((b1 - b2 + 1) * "1", 2)
    return ((arr >> b2) & width_int).astype('uint8')

def cloud_confidenceS3(arr):
    """
    00 = "Not Determined" = Algorithm did not determine the status of this condition
    01 = "No" = Algorithm has low to no confidence that this condition exists (0-33 percent confidence)
    10 = "Maybe" = Algorithm has medium confidence that this condition exists (34-66 percent confidence)
    11 = "Yes" = Algorithm has high confidence that this condition exists (67-100 percent confidence
    """
    return _capture_bitsS3(arr, 1, 0) #need to check these two integers are correct. See LS8 manual.

def pltshowimage(image):
    plt.imshow(image)
    plt.colorbar()
    plt.show()

def pltshowimageextra(image):
    plt.imshow(image)
    plt.colorbar()
    plt.clim(0, 14)
    plt.show()

# Plotting BT arrays between 285 K and 315 K 
def pltshowBTimages(image):
    plt.imshow(image)
    plt.colorbar()
    # plt.clim(292, 315)
    plt.clim(285, 296)
    # plt.axis('off')
    # plt.clim(289, 315)
    # plt.clim(300, 326)
    # plt.clim(293, 313)
    plt.show()

# def RMSE(array1,array2):
#     array1 = np.asarray(array1,dtype = np.float64)
#     array2 = np.asarray(array2,dtype = np.float64)
#     rmse =  math.sqrt(mean_squared_error(array1, array2))
#     return rmse 

def RMSE(array1,array2): 
    array1 = np.asarray(array1,dtype = np.float64)
    array2 = np.asarray(array2,dtype = np.float64)
    # return np.sqrt(((predictions - targets) ** 2).mean()) 
    return np.sqrt(((array1 - array2) ** 2).mean()) 

def RMedianSE(array1,array2): 
    array1 = np.asarray(array1,dtype = np.float64)
    array2 = np.asarray(array2,dtype = np.float64)
    square_error = (array1 - array2) ** 2
    MedianSqError = np.median(square_error)
    RMedianSqError = np.sqrt(MedianSqError)
    return RMedianSqError

# RMSE method that excludes any borders/pixels that are 0 
def RMSEfornonzeropixels(array1, array2): 
    array1 = np.asarray(array1,dtype = np.float64)
    array2 = np.asarray(array2,dtype = np.float64)
    return np.sqrt(((array1[array1>0] - array2[array2>0]) ** 2).mean())
    return rmse 

def RMedianSEfornonzeropixels(array1, array2): 
    square_error = (array1[array1>0] - array2[array2>0]) ** 2
    MedianSqError = np.median(square_error)
    RMedianSqError = np.sqrt(MedianSqError)
    return RMedianSqError

"*-------------- Start of Main --------------*"
"* -------------- Loading BT S3 and L8 Arrays --------------*"
refFilename = '/home/aaron/Documents/Python/MULTIMODAL_V1/Reference_L8_ThermalThermal_Test2.jpg'
sensedFilename = '/home/aaron/Documents/Python/MULTIMODAL_V1/Sensed_S3_ThermalThermal_Test2.jpg'


imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
imSensed = cv2.imread(sensedFilename, cv2.IMREAD_COLOR)

pltshowimage(imSensed)
pltshowimage(imReference)

reference_img = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)
sensed_img = cv2.cvtColor(imSensed, cv2.COLOR_BGR2GRAY)

pltshowimage(reference_img)
pltshowimage(sensed_img)


rmse_before = RMSE(sensed_img, reference_img)
print("RMSE Before:", rmse_before)

"*-------------- Extract SIFT Keypoints and Descriptors of Each Image --------------*"
print("*-------------- Extract SIFT Keypoints and Descriptors of Each Image --------------*")
# Second parameter is the binary mask for the L8/S3 image 
x_pixel_shift_arr = []
y_pixel_shift_arr = []

x_pixel_shift_H_tran_arr = []
y_pixel_shift_H_tran_arr = []

num_features_pre_ransac_arr = [] 
num_features_post_ransac_arr = [] 

mag_x_y_arr = [] 
euclidean_dist_arr = [] # Euclidean Distance of the attained X/Y-Pixel Shift after registration 
rmse_after_arr = []
rmse_after_H_tran_arr = []
count_num_nonbestcases = 0
count_num_bestcases = 0

xcoord_t0 = []
ycoord_t0 = []              
xcoord_s0 = []
ycoord_s0 = []

xcoord_t = []
ycoord_t = []
xcoord_s = []
ycoord_s = []

runtime_count = 0 

# Note: If you do not want to use a mask, set the mask to None. 
x_y_coordinates_source_kps, source_kps, descriptor_source = extract_SIFT_L8(sensed_img, None)  
x_y_coordinates_target_kps, target_kps, descriptor_target = extract_SIFT_L8(reference_img, None)

"*-------------- Match SIFT Descriptors --------------*"
matches_outliers, pos, matches_outliers2 = match_SIFT(descriptor_source, descriptor_target)
# print("Number of matches pre_RANSAC:", len(matches_outliers))

# Draw top matches pre-RANSAC. TO UNCOMMENT if you want to see matches pre-RANSAC
# img3 = cv2.drawMatches(uint8_S3_BT, source_kps, uint8_L8_BT, target_kps, matches_outliers2[:948], uint8_L8_BT, flags=2)
# plt.imshow(img3),plt.show()

# Compute RANSAC algorithm to eliminate outliers and Homography Matrix 
H, inlier_numbers, s, t, s0, t0 = myaffine_matrix(x_y_coordinates_source_kps,  x_y_coordinates_target_kps, pos)
num_matched_inliers = len(inlier_numbers[0])
print("Number of Matches Pre-RANSAC:", t0.shape[1])
print("Number of inliers in affine matrix fn:", num_matched_inliers)
# print("Indices of Chosen Inliers after RANSAC:", inlier_numbers)

num_features_pre_ransac_arr.append(t0.shape[1])
num_features_post_ransac_arr.append(num_matched_inliers)

x_pixel_shift_after_reg = H[0,2]
y_pixel_shift_after_reg = H[1,2]
tx = x_pixel_shift_after_reg
ty = y_pixel_shift_after_reg

x_pixel_shift_arr.append(x_pixel_shift_after_reg)
y_pixel_shift_arr.append(abs(y_pixel_shift_after_reg))

uint8_reference_copy = imReference.copy() # target image
uint8_sensed_copy = imSensed.copy() # source image

var = 0 
var2 = 0

for var2 in range(t0.shape[1]):
        xcoord_t0.append(t0[0][var2])
        ycoord_t0.append(t0[1][var2])                 
        xcoord_s0.append(s0[0][var2])
        ycoord_s0.append(s0[1][var2])


for var in range(num_matched_inliers):
        x_t = int(round(t[0][var]))
        y_t = int(round(t[1][var]))
        x_s = int(round(s[0][var]))
        y_s = int(round(s[1][var]))
        img_target = cv2.circle(uint8_reference_copy,(x_t, y_t), 3, (255,255,255), -1)   
        img_source = cv2.circle(uint8_sensed_copy,(x_s, y_s), 3, (255,255,255), -1)   
        
        xcoord_t.append(t[0][var])
        ycoord_t.append(t[1][var])                 
        xcoord_s.append(s[0][var])
        ycoord_s.append(s[1][var])


# reset
var = 0 
var2 = 0 

# Concatenate Horizontally the L8 and S3 images with Inliers 
matched_inliers_img = np.hstack((img_target,img_source))

"*-------------- Drawing the Matched Inliers --------------*"
for var in range(num_matched_inliers):
        x_t = int(round(t[0][var]))
        y_t = int(round(t[1][var]))
        x_s = int(round(s[0][var]))
        y_s = int(round(s[1][var]))
        cv2.line(matched_inliers_img, (x_t, y_t), (img_target.shape[1]+x_s, y_s), (255,255,255), 1)

# Draw lines between inliers in the Target(L8) and Source(S3) images
# cv2.line(matched_inliers_img, (int(t[0][0]), int(t[1][0])), ((cols)+int(s[0][0]), int(s[1][0])), (255,255,255), 1)
# np.save("/home/aaron/Documents/Python/IMG_REG_RANSAC_V37/Results/Saved_Images/matched_inliers_img_%d.npy"%runtime_count, matched_inliers_img)
pltshowimage(matched_inliers_img)

"*-------------- RMSE After Registration for Kelvin Images --------------*"
print("*-------------- RMSE After Registration for Kelvin Images --------------*")
# print(H)

# Deleting third row from the H matrix which is [0, 0, 1]
H = np.delete(H,(2), axis=0)
print(H, "\n")

new_tx, b = curve_fit(fit_func, xcoord_s, xcoord_t, 0)
new_ty, b2 = curve_fit(fit_func, ycoord_s, ycoord_t, 0)

new_tx = float(new_tx)
new_ty = float(new_ty)

print("New tx:", new_tx)
print("New ty:", new_ty)

H_trans = np.zeros((2,3))
H_trans[0,0] = 1
H_trans[0,1] = 0 
H_trans[0,2] = new_tx  
H_trans[1,0] = 0
H_trans[1,1] = 1
H_trans[1,2] = new_ty  
print(H_trans)

ceiled_tx = int(np.ceil(abs(tx)))
ceiled_ty = int(np.ceil(abs(ty)))

print("Abs Ceil value for tx:", ceiled_tx)
print("Abs Ceil value for ty:", ceiled_ty)

ceiled_new_tx = int(np.ceil(abs(new_tx)))
ceiled_new_ty = int(np.ceil(abs(new_ty)))


x_pixel_shift_H_tran_arr.append(new_tx)
y_pixel_shift_H_tran_arr.append(abs(new_ty))

# Reset to store only the x and y coordinates of the inliers of the next estimated homography 
xcoord_t = []
ycoord_t = []          
xcoord_s = []
ycoord_s = []

xcoord_t0 = []
ycoord_t0 = []              
xcoord_s0 = []
ycoord_s0 = []

# Aligned Downscaled/Upscaled BT image in Kelvin.  
sensed_img_reg = cv2.warpAffine(sensed_img.copy(), H, (cols, rows))
sensed_img_reg_H_trans = cv2.warpAffine(sensed_img.copy(), H_trans, (cols, rows))

reference_img2 = reference_img.copy()
reference_img2_duplicate = reference_img.copy()

pltshowimage(sensed_img_reg)
pltshowimage(sensed_img_reg_H_trans)

"*-------------- Conditions for Cropping to Compute RMSE After. Croopping is based on values of tx and ty --------------*"

rmse_after_H = RMSE(sensed_img_reg, reference_img2)
rmse_after_H_trans =  RMSE(sensed_img_reg_H_trans, reference_img2_duplicate)

print("RMSE After using H original:",  rmse_after_H)
print("RMSE After from H translation:",  rmse_after_H_trans, "\n")

rmse_after_arr.append(abs(rmse_after_H))
rmse_after_H_tran_arr.append(abs(rmse_after_H_trans))


print("Runtime: %s seconds ---" % (time.time() - start_time))


