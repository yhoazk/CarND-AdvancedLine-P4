#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from line import *
from lane import *
from img_proc import *
# define color boundaries
boundaries = [
    ([17, 15, 100], [50, 56, 200]),
    ([86, 31, 4], [220, 88, 50]),
    ([25, 146, 190], [62, 174, 250]),
    ([103, 86, 65], [145, 133, 128])
]


def get_yellow(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv, (20,50,150),(40,255,255))
    return mask

def remove_outliers(data_x, data_y, m=2):
    mean = np.mean(data_x)
    #print(mean)
    std_data = m*np.std(data_x)
    #print(std_data)
    #ret_data = [d for d in data if (abs(d-mean) < std_data) else mean]
    #ret_data = [d  if (abs(d-mean) < std_data) else mean for d in data]
    ret_data_y = []
    ret_data_x = []
    for x,y in zip(data_x, data_y):
        if (abs(x-mean) < std_data):
            ret_data_x.append(x)
            ret_data_y.append(y)


    #print(ret_data)
    return (ret_data_y, ret_data_x)

def birds_eye_transform(img):
    # from the center of the image
    print(img.shape)
    h,w = img.shape[:2]
    #pts = np.array([[w*0.19, h*.95], [(w*0.47), h*0.62], [w*0.53, h*0.62], [w*0.83, h*.95]], np.int32)
    #pts = np.array([[585, 460], [203,720], [1127, 720], [695, 460]], np.int32)
    #pts = np.array([[320, 0], [320, 720], [970, 720], [960, 0]], dtype='float32')
    #pts_rs = pts.reshape((-1,1,2))
    src_pts = np.array([[320, 0], [320, 720], [970, 720], [960, 0]], dtype='int32')
    # destination points for birds eye transformation
    #dst_pts = np.array([[585, 460], [203, 720], [1127, 720], [695, 460]], dtype='float32').reshape((-1, 1, 2))

    dst_pts = np.array([[577, 460], [240, 685], [1058, 685], [705, 460]], dtype='float32').reshape((-1, 1, 2))

    #poly_pts = np.int(src_pts)#.reshape((-1,1,2))
    img = cv2.polylines(img,[src_pts],True,(0,255,255), thickness=3)
    #
#   dst_pts =np.array([[(w / 4), 0],[(w / 4), h],[(w * 3 / 4), h],[(w * 3 / 4), 0]], dtype='float32')
    #dst_pts =np.array([[320, 0],[320, 720],[970, 720],[960, 0]], dtype='float32')
    #dst_pts = np.array([[585, 460], [203,720], [1127, 720], [695, 460]], dtype='float32').reshape((-1,1,2))
    #pts = np.float32(pts)
    mtx_bv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    bird_img = cv2.warpPerspective(img, mtx_bv, (w,h) )
   # bird_img = bird_img[::-1]
    return  bird_img


def rev_birds_eye_transform(img):
    # from the center of the image
    h,w = img.shape[:2]
    #pts = np.array([[w*0.19, h*.95], [(w*0.47), h*0.62], [w*0.53, h*0.62], [w*0.83, h*.95]], np.int32)
    pts = np.array([[585, 460], [203,720], [1127, 720], [695, 460]], np.int32)
    pts_rs = pts.reshape((-1,1,2))
    img = cv2.polylines(img,[pts_rs],True,(0,255,255))
#   dst_pts =np.array([[(w / 4), 0],[(w / 4), h],[(w * 3 / 4), h],[(w * 3 / 4), 0]], dtype='float32')
    dst_pts =np.array([[320, 0],[320, 720],[970, 720],[960, 0]], dtype='float32')
    pts = np.float32(pts)
    mtx_bv = cv2.getPerspectiveTransform(dst_pts, pts)
    warp_img = cv2.warpPerspective(img, mtx_bv, (w,h) )
   # bird_img = bird_img[::-1]
    return  warp_img

def get_hist_slice(img, slices=10, margin=100):
    """
    Returns the possible location of the center of the line
    based on the pixels with val = 1
    The image received has to be binarized.
    """
    h_img = img.shape[0]
    w_img = img.shape[1]
    location_l = []
    location_r = []
    location_ry = []
    location_ly = []

    """
    Createa a mask to ignore the center and the extremes of the image.
    ****-----***-----****
    ****-----***-----****
    ****-----***-----****
    ****-----***-----****
    ****-----***-----****
    ****-----***-----****
    ****-----***-----****
    """
    zero_patch = np.zeros((h_img, margin))
    one_patch  = np.ones((h_img, (w_img//2)-(1.5*margin)))

    mask = np.c_[zero_patch, one_patch]
    mask = np.c_[mask, zero_patch]
    mask = np.c_[mask, one_patch]
    mask = np.c_[mask, zero_patch]
    img = np.uint8(img)
    mask = np.uint8(mask)
    print("img:"+str(img.shape))
    print("mask:"+str(mask.shape))
    img = cv2.bitwise_and(img,img,mask = mask)

    for window in reversed(range(0,h_img, int(h_img/ slices))):
        sli = img[window:int(window+(h_img/slices)), :]
        sli_sum = np.sum(sli, axis=0)  # get the sum from all the columns
        """
        Add a margin to the histogram to not take pixels at the far left or right
        """
        sli_l, sli_r = (sli_sum[:w_img//2], sli_sum[w_img//2:])

        # get the location of 5 top max elements
        l_arg = np.argpartition(sli_l, -5)[-5:]
        r_arg = np.argpartition(sli_r, -5)[-5:]
        # Get the value of the max values and decide of this portion
        # of frame contains something interesiting
        mag_r = sum(sli_r[r_arg])
        mag_l = sum(sli_l[l_arg])
        if mag_l > 100:
            l_indx = np.mean(l_arg)
            location_l.append(l_indx)
            location_ly.append(window)

        if mag_r > 100:
            r_indx = np.mean(r_arg) + w_img//2
            location_ry.append(window)
            location_r.append(r_indx)
        #print("r_indx: " + str(r_indx) + " sli_r: " + str(sli_r))
        # add condtion for the case when the index is 0
    # if a point is 0 make its value the median btw the point before and the point after
    location = {'l':location_l, 'r':location_r, 'ly':location_ly, 'ry':location_ry}
    print("l : " + str(len(location_l)))
    print(location_l)
    print("ly : " + str(len(location_ly)))
    print(location_ly)

    print("r : " + str(len(location_r)))
    print(location_r)
    print("ry : " + str(len(location_ry)))
    print(location_ry)


    return location


def get_lane_area(poly_l, poly_r, n=10):
    """
    Create a list of points that describe the
    area between the lanes, starts from the bottom
    left corner
    """
    # set the list of points as needed by polyfill
    x = np.linspace(0, 720, n)
    fy_l = poly_l(x)
    fy_r = poly_r(x)

    return np.append(np.c_[fy_l,x], np.c_[fy_r,x][::-1], axis=0)

def process_lane(img):
    img_th = th_image(img)
    b_img = birds_eye_transform(img_th)
    #plt.imshow(b_img)
    lane_pts = get_hist_slice(b_img)
    #fit_l, v_l = np.polyfit(*remove_outliers(lane_pts['ly'], lane_pts['l']), deg=2,cov=True)
    #fit_r, v_r = np.polyfit(*remove_outliers(lane_pts['ry'], lane_pts['r']), deg=2,cov=True)
    #plt.scatter(lane_pts['l'], lane_pts['ly'], s=50, c='red', marker='o')
    #plt.scatter(lane_pts['r'], lane_pts['ry'], s=50, c='red', marker='o')
    try:
        fit_l, v_l = np.polyfit(*remove_outliers(lane_pts['l'], lane_pts['ly']), deg=2,cov=True)
    except:
        fit_l = np.polyfit(*remove_outliers(lane_pts['l'], lane_pts['ly']), deg=2)

# The covariance matrix yields to exceptions in certain scenarios, then is not a reliable estimator
    try:
        fit_r, v_r = np.polyfit(*remove_outliers(lane_pts['r'], lane_pts['ry']), deg=2,cov=True)
    except:
        fit_r = np.polyfit(*remove_outliers(lane_pts['r'], lane_pts['ry']), deg=2)
    poly_line_l = np.poly1d(fit_l)
    poly_line_r = np.poly1d(fit_r)
    #plt.plot(poly_line_l(lane_pts['ly']), lane_pts['ly'], 'g^')
    #plt.plot(poly_line_r(lane_pts['ry']), lane_pts['ry'], 'r^')
    #   plt.show()
    shade_polygon = np.int32(get_lane_area(poly_line_l, poly_line_r))

    shade = np.zeros_like(img_th)
    shade = cv2.fillConvexPoly(shade,shade_polygon,1)
    warp = rev_birds_eye_transform(shade)
    warp_3 = np.dstack([220*warp,150*warp,20*warp])
    shaded_lane= cv2.addWeighted(img, 0.6, warp_3, 0.4,0)
    return shaded_lane


def th_image(img):
    img_g = np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    s_img = img_g[:,:,2]
    ret, th = cv2.threshold(s_img,150,255,cv2.THRESH_BINARY)
    plt.imshow(th, cmap='gray')
    plt.show()
    #th_img = np.zeros_like(s_img)
    #th_img[(s_img > 50)] = 1
#    edge = cv2.Canny(s_img,250,200)
    #sbl_img = np.abs(cv2.Sobel(th_img, cv2.CV_64F, 0,1))
    return th

def main(img):
    im = camera.undistort(img)
    line_l.update(im)
    line_r.update(im)
    return  lane.process_lane(im)



if __name__ == '__main__':
    # create an instance for left and right lines
    line_l = Line('l')
    line_r = Line('r')
    # pass the instances to the lane constructor
    lane = Lane(line_l, line_r)
    camera = img_proc()
    camera.camera_calibration("/home/porko/workspace/nd_selfDrive/CarND-Advanced-Lane-Lines/camera_cal/calibration*")
    """
    im = plt.imread("test3.jpg")
    im = camera.undistort(im)
    im = main(im)
    plt.imshow(im)
    plt.show()
    """
    clip = VideoFileClip("../project_video.mp4")
    clip = clip.fl_image(main)
    clip.write_videofile("./out_video.mp4", audio=False)
    exit()
    files = glob("./sha*.jpg")
    for i in files:
        im = plt.imread(i)
        im = main(im)
        plt.imshow(im)
        plt.show()

        #img = plt.imread(i)
        #src_pts = np.array([[585, 460], [203,720], [1127, 720], [695, 460]], dtype='int32')
        #src_pts = np.array([[577, 460], [240, 685], [1058, 685], [705, 460]], dtype='int32')
        #pts_rs = src_pts.reshape((-1, 1, 2))
        #img = cv2.polylines(img, [pts_rs], True, (0, 255, 255), thickness=3)
        #plt.imshow(img)
        #plt.show()
        #th_m = camera.get_birdsView(img)

        #plt.imshow(th_m, cmap='gray')
        #plt.show()
        #th_m = th_image(th_m)
        #plt.imshow(th_m, cmap='gray')
        #plt.show()
        #img = main(img)

        #plt.imshow(img)
        #plt.show()
    #clip = VideoFileClip("../project_video.mp4")
    #clip = clip.fl_image(main)
    #clip.write_videofile("./out_video.mp4", audio=False)


#files = "test5.jpg"# glob("./*.jpg")
#files = glob("./*.jpg")
#files = "test_0044.jpg"# glob("./*.jpg")
#files = glob("./*.jpg")
#for i in files:
#    img = plt.imread(i)
#    img = process_lane(img)

#    cv2.imshow("h",img)
#    cv2.waitKey(400)
 #   cv2.destroyAllWindows()

#clip = VideoFileClip("../project_video.mp4")
#clip = clip.fl_image(process_lane)
#clip.write_videofile("./out_video.mp4", audio=False)
"""
    img_th = th_image(img)
    _,b = birds_eye_transform(img_th)
    img_zero = np.zeros_like(b)
    shade = np.zeros_like(b)
    lane_pts = get_hist_slice(b)
    #hist = np.sum(img[img.shape[0]/4:,:], axis=0)
    # b = th_image(b)
    ##### Find the lane fitting
    #fit_l = np.polyfit(lane_pts['ly'], remove_outliers(lane_pts['l'], 1.2), deg=2)
    #fit_r = np.polyfit(lane_pts['ry'], remove_outliers(lane_pts['r'], 1.2), deg=2)

    fit_l, v_l = np.polyfit(lane_pts['ly'], remove_outliers(lane_pts['l']), deg=2,cov=True)
    print("---v_l---")
    print(v_l)
    error_l = np.sum(np.abs(v_l[:][:][2]))

    fit_r, v_r = np.polyfit(lane_pts['ry'], remove_outliers(lane_pts['r']), deg=2,cov=True)
    print("---v_r----")
    print(v_r)
    error_r =np.sum(np.abs(v_r[:][:][2]))
    print("------------")
    poly_line_l = np.poly1d(fit_l)
    poly_line_r = np.poly1d(fit_r)
    print("error_r: " + str(error_r))
    print("error_l: " + str(error_l))
    line_lsp = np.linspace(0,720-1, 100)
    #line_l = fit_l[0]*fit_l[0]*fit_l[0]*line_lsp + fit_l[1]*fit_l[1]*line_lsp + fit_l[2]*line_lsp + lane_pts['l'][0]
    line_l = poly_line_l(line_lsp)
    line_r = poly_line_r(line_lsp)
    print("-----------------")
    print(fit_r)
    print(fit_l)
    #####

   # f, (ax1, ax2) = plt.subplots(1,2)
   # ax1.plot(lane_pts['ly'], lane_pts['l'])
    shade_polygon = np.int32(get_lane_area(poly_line_l, poly_line_r))
    shade = cv2.fillConvexPoly(shade,shade_polygon,1)
    warp = rev_birds_eye_transform(shade)
    warp_3 = np.dstack([120*warp,150*warp,20*warp])
    print("shape img" + str(img.shape))
    print("shape warp" + str(warp_3.shape))
    b = cv2.addWeighted(img, 0.6, warp_3, 0.4,0)
    plt.imshow(b)
#    plt.plot(line_l, line_lsp, 'g^')
#    plt.plot(line_r, line_lsp, 'r^')
#    plt.scatter(lane_pts['l'], lane_pts['ly'], s=50, c='red', marker='o')
#    plt.scatter(lane_pts['r'], lane_pts['ry'], s=50, c='red', marker='o')
    #plt.imshow(th_image(img))
    plt.show()
"""
