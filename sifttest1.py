#need to install opencv-contrib
#python3.5

import cv2
import random


img = cv2.imread('box.jpg')
r, c = img.shape[:2]
M = cv2.getRotationMatrix2D((c/2, r/2), 90, 1)
new_img = cv2.warpAffine(img, M, (c, r))
cv2.imwrite('image_rot.jpg', new_img)

img_rot = cv2.imread('image_rot.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_rot = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp, desc = sift.detectAndCompute(gray, None)
kp_rot, desc_rot = sift.detectAndCompute(gray_rot, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(desc, desc_rot, k=2)

good = []
for m, n in matches:
    if m.distance < 0.4 *n.distance:
        good.append([m])

random.shuffle(good)

image_match = cv2.drawMatchesKnn(img, kp, img_rot, kp_rot, good[:20], flags = 2, outImg = None)
cv2.imwrite('sift_matches.jpg', image_match)
