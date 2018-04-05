import numpy as np
import cv2


#NEED TO REVISE SIMPLY
def ave(sum, number):
    ave = sum / number
    return ave

PATH = 'C:\Code\objectrecognition\video'

tracker = cv2.TrackerKCF_create()
cam = cv2.VideoCapture('test_lr_1.mp4')
#cam.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, first_frame = cam.read() #train Image
gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
w_first, h_first = gray_first_frame.shape[::-1]


roi = cv2.imread('test_lr_1_roi.png') #query image
#res_roi = cv2.resize(roi, (int((1/8)*w_first), int((1/8)*h_first)), interpolation= cv2.INTER_CUBIC)
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

w, h = gray_roi.shape[::-1] #roi size 187 x 167

#initiate SIFT
sift = cv2.xfeatures2d.SIFT_create() #keypoint thresholod

#find keypoints and descriptors with SIFT
kp_roi, des_roi = sift.detectAndCompute(gray_roi, None)
kp_f_frm, des_f_frm = sift.detectAndCompute(gray_first_frame, None)

#FLANN parameters / Fast Library for Approximate Nearest Neighbors
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des_roi, des_f_frm, k=2)

#need to draw only good matches, create a mask
matchesmask = [[0,0] for i in range(len(matches))]

#save object point

good_key = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesmask[i] = [1,0]
        good_key.append(kp_f_frm[i].pt)

good_key = np.int32(good_key)

#good_key : (column/x/m , row/y/n)

max_r = 0
max_c = 0
min_r = 10000
min_c = 10000

for i, (m, n) in enumerate(good_key):

    if m >= max_c:
        max_c = m

    if n >= max_r:
        max_r = n

    if m <= min_c:
        min_c = m

    if n <= min_r:
        min_r = n


draw_params = dict( matchColor = (0, 255, 0), singlePointColor = (255, 0, 0), matchesMask = matchesmask, flags = 0)
img_res = cv2.drawMatchesKnn(gray_roi, kp_roi, gray_first_frame, kp_f_frm, matches, None, **draw_params)
cv2.imshow('draw_match', img_res)


sum_r = 0
sum_c = 0

for i in range(len(good_key)):
    sum_r += np.uint(good_key[i][1])
    sum_c += np.uint(good_key[i][0])

number = len(good_key)
ave_r = ave(sum_r, number)
ave_c = ave(sum_c, number)


middle_r = np.uint(ave_r)
middle_c = np.uint(ave_c)

width_first = int(max_c - min_c)
height_first = int(max_r - min_r)

track_window = (int(middle_c - 0.4 * width_first), int(middle_r - 0.4 * height_first), width_first, height_first)



ok = tracker.init(first_frame, track_window)

#initialize tracker with first frame and bounding box
h_save, w_save = first_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video_write = cv2.VideoWriter('test_lr.avi', fourcc, 25, (w_save, h_save))

count = 0

past_good_key_inter = []
temp_good_key_inter = good_key
gray_roi_inter = gray_roi

middle_c_inter = middle_c
middle_r_inter = middle_r

while (cam.isOpened()):
    ret, frame = cam.read()
    upd, obj = tracker.update(frame)

    #obj: ( 0: top column, 1: top row, 2: width(column), 3: height(row) )
    if upd: #draw bounding box
        x1 = (int(obj[0]), int(obj[1]))
        x2 = (int (obj[0] + 0.5 * obj[2] ), int(obj[1] + 0.5 * obj[3]))
        cv2.rectangle(frame, x1, x2, (255, 0, 0))
        #cv2.circle(frame, (int(obj[1] + 0.5 * obj[2]), int(obj[0] + 0.5 * obj[3])), 10, (0, 255, 0))
        #cv2.circle(frame, (middle_r_inter, middle_c_inter), 50, (255, 0, 0))

    video_write.write(frame)
    cv2.imshow("Track object", frame)

    count = count + 1

    if count % 30 == 0:
        print(count)
        #training frame
        gray_inter_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_inter_frm, des_inter_frm = sift.detectAndCompute(gray_inter_frame, None)
        matches_inter = flann.knnMatch(des_roi, des_inter_frm, k=2)
        matchesmask_inter = [[0, 0] for i in range(len(matches_inter))]

        good_key_inter = []
        try:
            for i_inter, (m_inter, n_inter) in enumerate(matches_inter):
                if m_inter.distance < 0.7 * n_inter.distance:
                    matchesmask_inter[i] = [1, 0]
                    good_key_inter.append(kp_inter_frm[i_inter].pt)
        except IndexError:
            good_key_inter = temp_good_key_inter



        good_key_inter = np.int32(good_key_inter)

        if len(good_key_inter) <= 3:
            good_key_inter = max_good_key_inter
            if len(max_good_key_inter) ==0:
                good_key_inter = good_key

        max_r_inter = 0
        max_c_inter = 0
        min_r_inter = 10000
        min_c_inter = 10000

        # good_key : (column/x/m , row/y/n)

        for i_inter, (m_inter, n_inter) in enumerate(good_key_inter):

            if m_inter >= max_c_inter:
                max_c_inter = m_inter

            if n_inter >= max_r_inter:
                max_r_inter = n_inter

            if m_inter <= min_c_inter:
                min_c_inter = m_inter

            if n_inter <= min_r_inter:
                min_r_inter = n_inter

        if min_c_inter == 10000:
            min_c_inter = 1

        if min_r_inter == 10000:
            min_r_inter = 1


        sum_r_inter = 0
        sum_c_inter = 0

        for i in range(len(good_key_inter)):
            sum_r_inter += np.uint(good_key_inter[i][1])
            sum_c_inter += np.uint(good_key_inter[i][0])

        number_inter = len(good_key_inter)
        ave_r_inter = ave(sum_r_inter, number_inter)
        ave_c_inter = ave(sum_c_inter, number_inter)


        middle_r_inter = np.uint(ave_r_inter)
        middle_c_inter = np.uint(ave_c_inter)

        width_inter = int(max_c_inter - min_c_inter + 1)
        height_inter = int(max_r_inter - min_r_inter + 1)

        track_window_inter = (int(middle_c - 0.4 * width_inter), int(middle_r_inter - 0.4 * height_inter), width_inter, height_inter)

        ok = tracker.init(frame, track_window_inter)

        #roi_inter
        roi_inter = frame[int(middle_r_inter - 0.4 * height_inter) : int(middle_r_inter + 0.4 * height_inter) , int(middle_c_inter - 0.4 * width_inter): int(middle_c_inter + 0.4 * width_inter)]
        gray_roi_inter = cv2.cvtColor(roi_inter, cv2.COLOR_BGR2GRAY)
        kp_roi, des_roi = sift.detectAndCompute(gray_roi_inter, None)

        #max good key
        if len(good_key_inter) >= len(past_good_key_inter):
            max_good_key_inter = good_key_inter
        past_good_key_inter = good_key_inter
        temp_good_key_inter = good_key_inter



        cv2.imshow('frame', gray_roi_inter)
        cv2.waitKey(1)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cam.release()
video_write.release()
cv2.destroyAllWindows()
