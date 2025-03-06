import numpy as np
import cv2 as cv
import os 

# Critères de précision pour l'affinage des coins
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Préparer les points de calibration 3D
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Stocker les points 3D et 2D
objpoints = []  # Points 3D réels
imgpoints = []  # Points 2D détectés sur l'image
#camera/imgs/img_calibration/cal2.png", "camera/imgs/img_calibration/cal2.png", "camera/imgs/img_calibration/cal1.png"
images = ["camera/imgs/pattern_tableH.png"]

for fname in images:
    #fname = "camera/imgs/img_calibration/"+fname
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Trouver les coins de l'échiquier
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Dessiner les coins détectés
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('Detected Corners', img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Calibration après collecte des points
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Affichage des matrices pour vérification
print("Matrice de la caméra :\n", mtx)
print("Coefficients de distorsion :\n", dist)

# Charger une nouvelle image à corriger
img = cv.imread("camera/imgs/pattern_tableH.png")
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Correction de la distorsion
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# Rogner l'image si nécessaire
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Affichage du résultat
cv.imshow("Undistorted Image", dst)
cv.waitKey(0)
cv.destroyAllWindows()
