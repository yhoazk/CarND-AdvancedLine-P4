import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os.path
import pickle

class img_proc():
    def __init__(self):
        # image shape
        self.w_img = 1280
        self.h_img = 720
        # source points for birds eye transformation
        self.src_pts = np.array([[320, 0],[320, 720],[970, 720],[960, 0]], dtype='float32')
        # destination points for birds eye transformation
        #self.dst_pts = np.array([[585, 460], [203,720], [1127, 720], [695, 460]], dtype='float32').reshape((-1,1,2))
        self.dst_pts = np.array([[577, 460], [240, 685], [1058, 685], [705, 460]], dtype='float32').reshape((-1,1,2))

        # birds eye matrix
        self.bv_matrix = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)
        # reversed birds eye matrix
        self.rev_bv_matrix = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        # The distorion coefficients
        self.dist_mtx = None
        self.dist_dist = None


    def camera_calibration(self, regexp):
        """
        regexp is the regular expression for the glob
        """

        # Number of corners in x
        nx = 9
        # Number of corners in y
        ny = 6
        # Pickle file path
        CAMERA_UNDISTORT_FILE = "/home/porko/workspace/nd_selfDrive/CarND-Advanced-Lane-Lines/output_images/calibration/"
        # Pickle file name
        CAMERA_PICKLE_NAME = "camera_undist.p"
        if os.path.isfile(CAMERA_UNDISTORT_FILE + CAMERA_PICKLE_NAME ):
            rest_cam = pickle.load(open(CAMERA_UNDISTORT_FILE + CAMERA_PICKLE_NAME,"rb") )
            self.dist_dist = rest_cam["dist"]
            self.dist_mtx = rest_cam["mtx"]
            return

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        imgs = glob("/home/porko/workspace/nd_selfDrive/CarND-Advanced-Lane-Lines/camera_cal/calibration*")

        img_list = [None] * len(imgs)
        for i, path_img in enumerate(imgs):
            img = plt.imread(path_img)
            g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(g_img, (nx, ny))

            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                img = cv2.drawChessboardCorners(img, (nx, ny), corners, True)
                img_list[i] = img.copy()

        # This function returns the camera matrix (mtx), distortion coefficients and rotation and translation vectors
        h, w = img_list[0].shape[:2]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h, w), None, None)
        self.dist_mtx = mtx
        self.dist_dist = dist
        camera_undist = {"mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}
        pickle.dump(camera_undist, open(CAMERA_UNDISTORT_FILE + CAMERA_PICKLE_NAME, "wb"))

    def get_birdsView(self, img):
        return cv2.warpPerspective(img, self.bv_matrix, (self.w_img,self.h_img) )

    def get_reverseBirdsView(self, img):
        return cv2.warpPerspective(img, self.rev_bv_matrix, (self.w_img,self.h_img) )

    def test_in(self):
        print("inheritance working")

    def undistort(self, img):
        return cv2.undistort(img, self.dist_mtx, self.dist_dist, None, self.dist_mtx)


