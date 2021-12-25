from sklearn.datasets import fetch_lfw_people
import numpy as np
import cv2

faces = fetch_lfw_people()
positive_patches = faces.images

i = 0
dist2 = cv2.normalize(positive_patches[1], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('1', dist2)

cv2.waitKey(0)