import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys


def ave(*args):
    sum = 0
    for i in args:
        sum = sum + i
    number = len(args)
    ave = sum / number
    return ave


tracker = cv2.TrackerKCF_create()

cam = cv2.VideoCapture('bbible.mp4')


roi = cv2.imread('bbible_roi_small.jpg')
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
ret, first_frame = cam.read()
gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
w, h = gray_roi.shape[::-1]

#initiate SIFT
sift = cv2.xfeatures2d.SIFT_create(10)

#find keypoints and descriptors with SIFT
kp_roi, des_roi = sift.detectAndCompute(gray_roi, None)
kp_f_frm, des_f_frm = sift.detectAndCompute(gray_first_frame, None)

#FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des_roi, des_f_frm, k=2)

print(len(matches))

#need to draw only good matches, create a mask
matchesmask = [[0,0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.6 * n.distance:
        matchesmask[i] = [1,0]


#draw_params = dict( matchColor = (0, 255, 0), singlePointColor = (255, 0, 0), matchesMask = matchesmask, flags = 0)

#img_res = cv2.drawMatchesKnn(gray_roi, kp_roi, gray_first_frame, kp_f_frm, matches, None, **draw_params)

for i in range(len(kp_f_frm)):
    sum_r = 0
    sum_c = 0
    sum_r += kp_f_frm[i].pt[0]
    sum_c += kp_f_frm[i].pt[1]


w, h = gray_roi.shape[::-1]

ave_r = ave(sum_r)
ave_c = ave(sum_c)
top_r = np.uint(ave_r - (1/2)*w)
top_c = np.uint(ave_c - (1/2)*h)

top = (top_r, top_c)
bottom = (top_r + w, top_c + h)

track_window = (top_c, top_r, w, h)


ok = tracker.init(first_frame, track_window) #initialize tracker with first frame and bounding box

while (cam.isOpened()):
    ret, frame = cam.read()
    upd, obj = tracker.update(frame)



    if upd: #draw bounding box
        x1 = (int(obj[0]), int(obj[1]))
        x2 = (int(obj[0] + obj[2]), int(obj[1] + obj[3]))
        cv2.rectangle(frame, x1, x2, (255, 0, 0))

    cv2.imshow("Track object", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
