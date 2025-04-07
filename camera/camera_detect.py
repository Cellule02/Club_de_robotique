import cv2
import cv2.aruco as aruco
import time
import numpy as np 
import matplotlib.pyplot as plt 

from vect import corrigeDeformation

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detectorParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detectorParams)



def get_center(pos):
    pos = pos[0]
    x = [float(i[0]) for i in pos]
    y = [float(i[1]) for i in pos]

    center_x = int(min(x) + (max(x)-min(x))/2)
    center_y = int(min(y) + (max(y)-min(y))/2)
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
            data_center[str(ids[i])] = get_center(marker_corners[i])
            data[str(ids[i])] = marker_corners[i]
        #data_center = eq_xy(data_center)
        return data_center
    except AttributeError:
        pass


def draw_object(img,data):

    for key,value in data.items():
            #print(key,"value is ", value)
            cv2.circle(img,value,1,(255,0,0),2)
            """cv2.line(img, [int(i) for i in value[0]],[int(i) for i in value[1]],(255,0,0),thickness=2)
            cv2.line(img, [int(i) for i in value[0]],[int(i) for i in value[3]],(255,0,0),thickness=2)
            cv2.line(img, [int(i) for i in value[1]],[int(i) for i in value[2]],(255,0,0),thickness=2)
            cv2.line(img, [int(i) for i in value[2]],[int(i) for i in value[3]],(255,0,0),thickness=2)"""

            cv2.putText(img,str(key),(value[0]-10,value[1]-10),cv2.FONT_HERSHEY_COMPLEX,1,color=(0,255,0),thickness=2)

    cv2.imshow("test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_gradin(img):
    HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    H,S,V = cv2.split(HSV)
    #R,G,B = cv2.split(img)
    ig2= cv2.equalizeHist(grey)
    _,thresh = cv2.threshold(ig2,240,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.dilate(thresh,kernel=kernel, iterations=1)
    thresh = cv2.medianBlur(thresh,5)

    cv2.imshow("t", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("camera\imgs\img_test\\test6_screenshot_20.02.2025.png")
detect_gradin(img)

# Process the image and draw markers
url = "/dev/video0"

"""cap = cv2.VideoCapture(url)


while True:
    ret, frame = cap.read()

    if ret == True:
        data = get_aruco_id(frame)
        print(data)
        draw_object(frame,data)
        #get_center(data[22])
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

img = cv2.imread("camera\imgs\img_test\\test1_screenshot_20.02.2025.png")
#print(img.shape) 480x640
center = get_aruco_id(img)

#corrigeDeformation(QR1_Virt,QR2_Virt,QR3_Virt,QR4_Virt,QR_Robot_Virt)
"""print("coordonnée en de qr2",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["20"]))
print("coordonnée en de q1",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["21"]))
print("coordonnée en de q4",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["22"]))
print("coordonnée en de q3",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["23"]))"""

#print("coordonnée en de q3",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["22"]))

#draw_object(img=img,data=center)



