import cv2
import numpy as np
from pathlib import Path
from pprint import pprint

row = 7 # 行の数
col = 10 # 列の数
corner_num = (col, row) # コーナーの数
size = 3 # コーナー間の距離

pW = np.empty([row * col, 3], dtype=np.float32)
for i_row in range(row):
    for i_col in range(col):
        pW[i_row* col + i_col] = \
                np.array([size * i_col, size * i_row, 0], dtype=np.float32)

pWs = []
qIs = []

for path_img in Path('.').glob('*.JPG'):
    print('Read {}...'.format(str(path_img)))
    img = cv2.imread(str(path_img), 0)
    found, qI = cv2.findChessboardCorners(img, corner_num)

    if found: #コーナー検出成功
        print('success to find corner!!')
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        qI_sub = cv2.cornerSubPix(img, qI, (5, 5), (-1, -1), term)
        pWs.append(pW)
        qIs.append(qI_sub)

    else: #コーナー検出失敗
        print('fail to find corner!!')
        continue

rep, K, d, rvec, tvec = cv2.calibrateCamera(pWs, qIs,
        (img.shape[1], img.shape[0]), None, None)
print('Internal camera parameter: K')
pprint(K)
np.save('K.npy', K)
np.save('d.npy', d)
