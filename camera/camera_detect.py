import cv2 as cv
import cv2.aruco as aruco
import time
import numpy as np 
import matplotlib.pyplot as plt 
import glob

from vect import corrigeDeformation

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detectorParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detectorParams)


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
    return mtx,dist

def get_center(pos):
    #print(pos)
    pos = pos[0]
    
    x = [float(i[0]) for i in pos]
    y = [float(i[1]) for i in pos]

    center_x = int((max(x)+min(x))/2)
    center_y = int((max(y)+min(y))/2)
    #print(center_x,center_y)
    return center_x, center_y

def eq_xy(qr_center):
    x22_x20 = int((qr_center["22"][0] + qr_center["20"][0])/2)
    x23_x21 = int((qr_center["23"][0] + qr_center["21"][0])/2)

    y21_y20 = int((qr_center["21"][1]+qr_center["20"][1])/2)
    y23_y22 = int((qr_center["23"][1]+qr_center["22"][1])/2)

    mean_align_center={
        "20":(x22_x20,y21_y20),
        "21":(x23_x21,y21_y20),
        "22":(x22_x20,y23_y22),
        "23":(x23_x21,y23_y22)
    }
    return mean_align_center


def get_aruco_id(img, detect=detector):
    try:
        data_center = {}
        data = {}
        marker_corners, marker_ids, rejected_candidates = detect.detectMarkers(img)
        ids = marker_ids.transpose()[0]
        for i in range(len(marker_ids)):
            #print("marker corner i",marker_corners[i])
            data_center[str(ids[i])] = get_center(marker_corners[i])
            data[str(ids[i])] = marker_corners[i]
        return data, data_center
    except AttributeError as e:
        print(e)


def draw_object(img,data):

    if (type(data) == dict):
        for key,value in data.items():
                #print(key,"value is ", value)
                value = value[0]
                #print(value)
                #cv.circle(img,value,1,(255,0,0),2)
                cv.line(img, [int(i) for i in value[0]],[int(i) for i in value[1]],(0,255,0),thickness=2)
                cv.line(img, [int(i) for i in value[2]],[int(i) for i in value[3]],(0,255,0),thickness=2)
                cv.line(img, [int(i) for i in value[0]],[int(i) for i in value[3]],(0,255,0),thickness=2)
                cv.line(img, [int(i) for i in value[2]],[int(i) for i in value[1]],(0,255,0),thickness=2)

                cv.putText(img,str(key),[int(i -10) for i in value[3]],cv.FONT_HERSHEY_COMPLEX,1,color=(0,255,0),thickness=2)
    else:
        for value in data:
                #print("value is ", value)
                #cv.circle(img,value,1,(255,0,0),2)
                cv.line(img, [int(i) for i in value[0]],[int(i) for i in value[1]],(0,255,0),thickness=2)
                cv.line(img, [int(i) for i in value[2]],[int(i) for i in value[3]],(0,255,0),thickness=2)
                cv.line(img, [int(i) for i in value[0]],[int(i) for i in value[3]],(0,255,0),thickness=2)
                cv.line(img, [int(i) for i in value[2]],[int(i) for i in value[1]],(0,255,0),thickness=2)

                cv.putText(img,"g",[int(i -10) for i in value[3]],cv.FONT_HERSHEY_COMPLEX,1,color=(0,255,0),thickness=2)
    return img 

def show_img(img):
    cv.imshow("test",img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def verif_gradin(gradins):
    true_gradins = []
    for gradin in gradins:
        #print("gradin",gradin[:,0])
        x = gradin[:,0]
        y = gradin[:,1]
        #print('x',x)
        #print('y',y)
        
        distx = (max(x)-min(x))**2
        disty= (max(y)-min(y))**2

        if (distx/disty >= 2) or (disty/distx >=2):
            true_gradins.append(gradin)
            print("trouvé")
    return true_gradins

        
def detect_gradin(img):
    grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    ig2= cv.equalizeHist(grey)

    _,thresh = cv.threshold(ig2,240,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C)

    kernel = np.ones((11,7),np.uint8)
    
    thresh = cv.dilate(thresh,kernel=kernel, iterations=1)
    thresh = cv.medianBlur(thresh,3)
    thresh = cv.erode(thresh,kernel=kernel, iterations=1)
    thresh = 255-thresh
    contours,_ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #print(contours)

    gradins = [] #np.array([[305,185], [306,203], [451,109], [409,100]], dtype=np.int32)
    for cnt in contours:
        # Approximate the contour
        epsilon = 0.02 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        
        # If it has 4 points and is convex, it's likely a rectangle
        if len(approx) == 4 and cv.isContourConvex(approx):
            gradins.append(approx[:,0])
            #print("approx",approx[:,0])
            # Draw the rectangle
            cv.drawContours(img, [approx], 0, (0, 255, 0), 2)
    gradins = verif_gradin(gradins)
    gradins_center = []
    for i in gradins:
        #print('i')
        gradins_center.append(get_center([i]))
    show_img(contours)
    #print("gradin",gradins)
    return gradins, gradins_center


def is_colision(vendengeuse, enemy, range=50):
    dist = ((vendengeuse[0]-enemy[0])**2 + (vendengeuse[1]+enemy[1])**2)**0.5
    if (dist < range):
        return True
    else:
        return False
    

def get_bluebot(acuro_pos):
    for key in acuro_pos.keys():
        key = int(key)
        if (key >=51) and (key <=70):
            return acuro_pos[str(key)]
        else:
            print("erreur blue")
        
def get_yellowbot(acuro_pos):
    for key in acuro_pos.keys():
        key=int(key)
        if (key >=71) and (key<=90):
            return acuro_pos[str(key)]
        else:
            print("erreur jaune")


def get_vendengeuse(color, pos):
    bots = [get_bluebot(pos), get_yellowbot(pos)]
    if (color=="blue"):
        return bots
    else:
        return bots[:,:,-1]


#print("notre robot ",get_vendengeuse("blue",center))
#img_obgj_detect=draw_object(img, gradins)
#print(corrigeDeformation(center["21"],center["20"],center["23"],center["22"],gradins_center[0]))
#img_obgj_detect2=draw_object(img_obgj_detect, data)
#show_img(img_obgj_detect)
# Process the image and draw markers

mtx, dist = calcam("camera/cprb1/*.jpg")
print(mtx,dist)
img = cv.imread('camera/imgs/test2/test_screenshot_16.04.20250.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
"""url = "/dev/video2"

cap = cv.VideoCapture(url)
"""

"""while True:
    ret, frame = cap.read()

    if ret == True:
        #data = get_aruco_id(frame)
        #print(data)
        #draw_object(frame,data)
        #get_center(data[22])
        show_img(frame)
    else:
        print("pas d'img")
        break"""


"""Pt_QR1_Reel = (60 , 60 ) # Pt_QR1_Virt Position (X:int ,Y:int) du QRCODE 1 sur le terrain (en cm)
Pt_QR2_Reel = (240, 60 ) # Pt_QR2_Virt Position (X:int ,Y:int) du QRCODE 2 sur le terrain (en cm)
Pt_QR3_Reel = (60, 140) # Pt_QR3_Virt Position (X:int ,Y:int) du QRCODE 3 sur le terrain (en cm)
Pt_QR4_Reel = (240, 140) # Pt_QR4_Virt Position (X:int ,Y:int) du QRCODE 4 sur le terrain (en cm)
"""
############ 300 cm #############
#                               # 2
#        QR3          QR4       # 0
#                               # 0
#        QR1          QR2       # c
#(0,0)                          # m
#################################

#QR1=21
#QR2=20
#QR3=23
#QR4=22

"""
#corrigeDeformation(QR1_Virt,QR2_Virt,QR3_Virt,QR4_Virt,QR_Robot_Virt)
print("coordonnée en de qr2",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["20"]))
print("coordonnée en de q1",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["21"]))
print("coordonnée en de q4",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["22"]))
print("coordonnée en de q3",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["23"]))"""

#print("coordonnée en de q3",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["22"]))

#draw_object(img=img,data=center)



