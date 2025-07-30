import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def averaging(img):
    kernel = np.ones((5, 5), np.float32) / 25 
    dst = cv.filter2D(img, -1, kernel)

    return dst

def bilateral_filtering(img):
    return cv.bilateralFilter(img, 9, 75, 75)

if __name__ == '__main__':
    img = cv.imread('noisy_image.png')
    assert img is not None, "file could not be read, check with os.path.exists()"

    # dst = averaging(img)
    dst = bilateral_filtering(img)

    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()


