import numpy as np
import cv2
from pathlib import Path
from pprint import pprint

row = 7
col = 10
corner_num = (col, row)
size = 2

# load internal parameter
K = np.load('K.npy')
d = np.load('d.npy')

pW = np.empty([row * col, 3], dtype=np.float32)
for i_row in range(row):
    for i_col in range(0, col):
        pW[i_row* col + i_col] = \
                np.array([size * i_col, size * i_row, 0], dtype=np.float32)

ex_img = cv2.imread('ex_img.jpg', 0)
found, qI = cv2.findChessboardCorners(ex_img, corner_num)

if found:
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    qI_sub = cv2.cornerSubPix(ex_img, qI, (5, 5), (-1, -1), term)

    ret, rvec, tvec = cv2.solvePnP(pW, qI, K, d)
    rmat = cv2.Rodrigues(rvec)[0]
    
    print('R')
    pprint(rmat)
    print('t')
    pprint(tvec)
