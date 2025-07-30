import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


if __name__ == '__main__':
    img = cv.imread('tsukuba.png', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    equ = cv.equalizeHist(img)
    res = np.hstack((img, equ))
    cv.imwrite('res.png', res)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    res2 = np.hstack((img, cl1))
    cv.imwrite('clahe_2.jpg', res2)