import numpy as np
import cv2 as cv
from cv2.typing import MatLike
from matplotlib import pyplot as plt

def water_shed(image: MatLike) -> tuple[float, MatLike]:
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    return (ret, thresh)

def segment_with_edge(image: MatLike) -> MatLike:

    blur_img = cv.blur(image, (4, 4))
    # edges = cv.cvtColor(blur_img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(blur_img, 100, 200)
    ret, edges = cv.threshold(edges,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    return edges

if __name__ == '__main__':

    image = cv.imread('images/water_coins.jpg')
    assert image is not None, "file could not be read, check with os.path.exists()"
    
    print(image.shape)
    original_height, original_width = image.shape[:2]
    new_width = 350
    ratio = new_width / original_width
    new_height = int(ratio * original_height)

    resized_image = cv.resize(image, (new_width, new_height))

    # ret, result = water_shed(resized_image)
    result = segment_with_edge(resized_image)

    cv.imshow("segmentation", result)

    cv.waitKey()

