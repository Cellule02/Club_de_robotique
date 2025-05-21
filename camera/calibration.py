import numpy as np
import cv2 as cv
import glob 

def calcam(img_folder):
        # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    img_size = None

    images = glob.glob(img_folder)
    print(len(images))
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_size=gray.shape[::-1]
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,5), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            #print("\n\n",corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,5), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    
        cv.destroyAllWindows()
    ### A TESTER transformer fx fy en mm 
    """print(gray.shape)
    print("mtx\n",mtx)
    print("dist\n",dist)
    print("rvecs\n",rvecs)
    print("tvecs\n",tvecs)"""
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx,dist,rvecs,tvecs


def reverse_perspective(img, mtx, rvec):
    # Obtenir les dimensions de l'image (attention à l'ordre)
    height, width, _ = img.shape  # img.shape donne (hauteur, largeur, canaux)
    
    # Convertir le vecteur de rotation en matrice de rotation
    R, _ = cv.Rodrigues(rvec)  # Notez qu'on récupère aussi un jacobien qu'on ignore
    
    # Calculer la matrice de correction (inverse de la rotation)
    R_correction = np.linalg.inv(R)
    
    # Construire l'homographie
    H = mtx @ R_correction @ np.linalg.inv(mtx)
    
    # Appliquer la transformation
    result = cv.warpPerspective(img, H, (width, height))
    
    return result

def reverse_perspective(img, mtx, rvec):
    # Obtenir les dimensions de l'image originale
    height, width, _ = img.shape
    
    # Convertir le vecteur de rotation en matrice de rotation
    R, _ = cv.Rodrigues(rvec)
    
    # Calculer la matrice de correction (inverse de la rotation)
    R_correction = np.linalg.inv(R)
    
    # Construire l'homographie de base
    H_base = mtx @ R_correction @ np.linalg.inv(mtx)
    
    # Définir les quatre coins de l'image originale
    corners = np.array([
        [0, 0],             # Coin supérieur gauche
        [width - 1, 0],     # Coin supérieur droit
        [width - 1, height - 1],  # Coin inférieur droit
        [0, height - 1]     # Coin inférieur gauche
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    # Transformer ces coins avec l'homographie de base
    # cv2.perspectiveTransform attend des points au format (n, 1, 2)
    transformed_corners = cv.perspectiveTransform(corners, H_base)
    
    # Trouver les coordonnées minimales et maximales après transformation
    min_x = np.min(transformed_corners[:, 0, 0])
    min_y = np.min(transformed_corners[:, 0, 1])
    max_x = np.max(transformed_corners[:, 0, 0])
    max_y = np.max(transformed_corners[:, 0, 1])
    
    # Calculer les nouvelles dimensions
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))
    
    # Créer une matrice de translation pour déplacer l'image dans le cadre visible
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])
    
    # Combiner la translation avec l'homographie de base
    H_final = T @ H_base
    
    # Appliquer la transformation perspective avec les nouvelles dimensions
    result = cv.warpPerspective(img, H_final, (new_width, new_height))
    
    return result

def img_undisort(img, mtx,dist):
    
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

mtx,dist,rvecs,tvecs = calcam("camera/cprb2/*.jpg")
img = cv.imread("camera/imgs/test2/test_screenshot_16.04.2025.png")
dst=img_undisort(img, mtx=mtx, dist=dist)
print(dst.shape)
cimg = reverse_perspective(dst,mtx,rvecs[0])
cv.imwrite("noperspective.jpg", cimg)


