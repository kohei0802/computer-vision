import cv2 as cv
import numpy as np,sys

if __name__ == '__main__':
    A = cv.imread('images/apple.png')
    B = cv.imread('images/orange.png')
    assert A is not None, "file could not be read, check with os.path.exists()"
    assert B is not None, "file could not be read, check with os.path.exists()"

    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpA.append(G)

    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpB.append(G) 

    lpA = [gpA[5]]
    for i in range(5, 0, -1): 
        GE = cv.pyrUp(gpA[i])
        # Resize to match the target size
        GE = cv.resize(GE, (gpA[i-1].shape[1], gpA[i-1].shape[0]))
        L = cv.subtract(gpA[i-1], GE)
        lpA.append(L) 

    lpB = [gpB[5]] 
    for i in range(5, 0, -1):
        GE = cv.pyrUp(gpB[i])
        # Resize to match the target size
        GE = cv.resize(GE, (gpB[i-1].shape[1], gpB[i-1].shape[0]))
        L = cv.subtract(gpB[i-1], GE)
        lpB.append(L) 

    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
        LS.append(ls)

    ls_ = LS[0]
    for i in range(1,6):
        ls_ = cv.pyrUp(ls_)
        # Also need to resize here for the reconstruction
        ls_ = cv.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv.add(ls_, LS[i])

    real = np.hstack((A[:,:cols//2],B[:,cols//2:]))

    cv.imwrite('images/Pyramid_blending2.jpg',ls_)
    cv.imwrite('images/Direct_blending.jpg',real)