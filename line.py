import numpy as np
from  img_proc import *
import cv2

class Line(img_proc):
    def __init__(self, side='l', debug=False):
        super().__init__()
        # was the line detected in the last iteration?
        self.detected = False
        # Number of frames to save
        self.FRAMES = 6
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #polynomial coefficients for the most recent fit in meters
        self.current_fit_scaled = [np.array([False])]
        #polynomial coefficients for the most recent fit in meters
        self.last_fit_scaled = [np.array([False])]
        #polynomial coefficients for the last fit
        self.last_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # a flag to indicate if mesasges and images are to be shown to debug
        self.debug = debug
        # determines if the lone is left line or right line_base_pos
        self.side = side
        # confidence
        self.confidence = 0
        # Error threshold
        self.poly = None
        self.last_poly = []
        self.y_pxm = 30/720
        self.x_pxm = 3.70/700
        # Buffer of curves to fillter
        self.buffer_curve = []

    def get_CurveRad(self):
        """
        :param poly:
        :return:
        """
        return np.mean(self.buffer_curve)

    def __get_ThresholdImg(self, img):
        img_g = np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
        l_img = img_g[:,:,1]
        s_img = img_g[:, :, 2]
        """
        Detect the color of the ground, if its clear, the l-channel does a better job
        if its dark the S channel its better
        in order to differentiate them, select a column and summ the thresholded values
        if the sum is large for an specific channel then choose the other one
        """
        ls = cv2.Sobel(l_img,cv2.CV_64F, 1,0,None,3)
        ls_abs = np.abs(ls)
        sb_t = np.zeros_like(ls_abs, dtype=np.uint8)
        sb_t[(ls_abs >= 150)&(ls_abs <= 255)] = 255
        #plt.imshow(ls_abs, cmap='gray')
        #plt.show()


        ret, th_s = cv2.threshold(s_img,120,255,cv2.THRESH_BINARY)
        ret, th_l = cv2.threshold(l_img,120,255,cv2.THRESH_BINARY)
        patch_1 = th_l[100:500:2, 600:640:2]

        sum_patch_l = np.sum(patch_1)

        # select the channel to use
        if sum_patch_l > 10000:
            result_image = th_s
        else:
            result_image = th_l


        #sth = cv2.adaptiveThreshold(s_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


        #th_img = np.zeros_like(s_img)
        #th_img[(s_img > 50)] = 1
        #edge = cv2.Canny(s_img,250,200)
        #sbl_img = np.abs(cv2.Sobel(th_img, cv2.CV_64F, 0,1))
        return result_image

    def remove_outliers(self, data_x, data_y, m=2):
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

        return ret_data_y, ret_data_x

    def __get_hist_slice(self, img, slices=10, margin=140):
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
        """
        zero_patch = np.zeros((h_img, margin))
        one_patch  = np.ones((h_img, (w_img//2)-(1.5*margin)))

        mask = np.c_[zero_patch, one_patch]
        mask = np.c_[mask, zero_patch]
        mask = np.c_[mask, one_patch]
        mask = np.c_[mask, zero_patch]
        img = np.uint8(img)
        mask = np.uint8(mask)
        # apply mask to entire image
        img = cv2.bitwise_and(img,img,mask = mask)
        if self.debug:
            plt.imshow(img)
            plt.show()

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
        if self.debug == True:
            print("l : " + str(len(location_l)))
            print(location_l)
            print("ly : " + str(len(location_ly)))
            print(location_ly)

            print("r : " + str(len(location_r)))
            print(location_r)
            print("ry : " + str(len(location_ry)))
            print(location_ry)

        # add the located points in the array
        if len(self.recent_xfitted) >= self.FRAMES:
            self.recent_xfitted.pop(0) # remove the oldest element
        # add the newest element to the queue
        self.recent_xfitted.append(location[self.side]) # add the newest values of x

        # temp
#        if side == 'l':
        return location

    def update(self, img):
        b_img  = self.get_birdsView(img)
        #plt.imshow(b_img)
        #plt.show()
        b_img = self.__get_ThresholdImg(b_img)
        #plt.imshow(b_img, cmap='gray')
        #plt.show()
        lane_pts = self.__get_hist_slice(b_img, margin=100)

        x,y = self.remove_outliers(lane_pts[self.side], lane_pts[self.side+'y'])
        x_sc,y_sc = self.remove_outliers(lane_pts[self.side], lane_pts[self.side+'y'])
        x_sc = np.asarray(x_sc) * self.x_pxm
        y_sc = np.asarray(y_sc) * self.y_pxm

        if len(lane_pts[self.side]) > 2:
            # Find the polynomial
            try:
                fit, v  = np.polyfit(x,y, deg=2, cov=True)
                fit_scaled, v_scaled  = np.polyfit(x_sc, y_sc, deg=2, cov=True)

            except:
                fit = np.polyfit(x, y, deg=2)
                fit_scaled= np.polyfit(x_sc, y_sc, deg=2)

                v = None
                v_scaled = None
            self.last_fit = self.current_fit
            self.current_fit = fit

            self.last_fit_scaled = self.current_fit_scaled
            self.current_fit_scaled = fit_scaled
        else:
            # use the past fit as we do not have enough information to decide
            self.current_fit = self.last_fit
            fit = self.last_fit
            v = None
        # if v == None then there was an exception in fitting the polynomial
        if v == None:
            # The covariance matrix could not be obtained, compare with previous
            # fit
            self.poly = np.poly1d(fit)
        else:
            # v indicates the error in fitting hte line, if its big
            # the confidence drops
            error_r = np.sum(np.abs(v[:][:][2]))
            # decide if add or not te new values


            # if error is small add it as best fit
            curverad = ((1 + (2 * fit_scaled[0] * 719 * self.y_pxm + fit_scaled[1]) ** 2) ** 1.5) / np.absolute(
                2 * fit_scaled[0])


            test_poly = np.poly1d(fit_scaled)
            y_test = np.linspace(0,720, 100)
            self.radius_of_curvature = curverad
            # filter by averaging
            self.buffer_curve.append(curverad)
            if len(self.buffer_curve) >= self.FRAMES:
                self.buffer_curve.pop(0)

            self.poly = np.poly1d(fit)
            if len(self.last_poly) > self.FRAMES:
                self.last_poly.pop(0)

            self.last_poly.append(self.poly)
        return b_img

    def get_LinePoly(self):
        """
        This returns the filtered polynomial
        To be used in the lane class
        """
        if len(self.last_poly) <= 3:
            # the first element
            self.best_fit = self.poly
        else:
            coffs = [x.coeffs for x in self.last_poly]
            coffs = np.mean(coffs, axis=0)


            self.best_fit = np.poly1d(coffs)
        return self.best_fit
