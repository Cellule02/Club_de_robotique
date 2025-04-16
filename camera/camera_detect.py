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
                #cv2.circle(img,value,1,(255,0,0),2)
                cv2.line(img, [int(i) for i in value[0]],[int(i) for i in value[1]],(0,255,0),thickness=2)
                cv2.line(img, [int(i) for i in value[2]],[int(i) for i in value[3]],(0,255,0),thickness=2)
                cv2.line(img, [int(i) for i in value[0]],[int(i) for i in value[3]],(0,255,0),thickness=2)
                cv2.line(img, [int(i) for i in value[2]],[int(i) for i in value[1]],(0,255,0),thickness=2)

                cv2.putText(img,str(key),[int(i -10) for i in value[3]],cv2.FONT_HERSHEY_COMPLEX,1,color=(0,255,0),thickness=2)
    else:
        for value in data:
                #print("value is ", value)
                #cv2.circle(img,value,1,(255,0,0),2)
                cv2.line(img, [int(i) for i in value[0]],[int(i) for i in value[1]],(0,255,0),thickness=2)
                cv2.line(img, [int(i) for i in value[2]],[int(i) for i in value[3]],(0,255,0),thickness=2)
                cv2.line(img, [int(i) for i in value[0]],[int(i) for i in value[3]],(0,255,0),thickness=2)
                cv2.line(img, [int(i) for i in value[2]],[int(i) for i in value[1]],(0,255,0),thickness=2)

                cv2.putText(img,"g",[int(i -10) for i in value[3]],cv2.FONT_HERSHEY_COMPLEX,1,color=(0,255,0),thickness=2)
    return img 

def show_img(img):
    cv2.imshow("test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ig2= cv2.equalizeHist(grey)

    _,thresh = cv2.threshold(ig2,240,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    kernel = np.ones((11,7),np.uint8)
    
    thresh = cv2.dilate(thresh,kernel=kernel, iterations=1)
    thresh = cv2.medianBlur(thresh,3)
    thresh = cv2.erode(thresh,kernel=kernel, iterations=1)
    thresh = 255-thresh
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)

    gradins = [] #np.array([[305,185], [306,203], [451,109], [409,100]], dtype=np.int32)
    for cnt in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # If it has 4 points and is convex, it's likely a rectangle
        if len(approx) == 4 and cv2.isContourConvex(approx):
            gradins.append(approx[:,0])
            #print("approx",approx[:,0])
            # Draw the rectangle
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
    gradins = verif_gradin(gradins)
    gradins_center = []
    for i in gradins:
        #print('i')
        gradins_center.append(get_center([i]))
    
    #print("gradin",gradins)
    return gradins, gradins_center


img = cv2.imread(r"camera\imgs\img_test\test_screenshot_20.02.2025.png")
data,center = get_aruco_id(img)
gradins, gradins_center = detect_gradin(img)
print("center",gradins_center)
img_obgj_detect=draw_object(img, gradins)
print(corrigeDeformation(center["21"],center["20"],center["23"],center["22"],gradins_center[0]))
img_obgj_detect2=draw_object(img_obgj_detect, data)
show_img(img_obgj_detect)
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

"""
#corrigeDeformation(QR1_Virt,QR2_Virt,QR3_Virt,QR4_Virt,QR_Robot_Virt)
print("coordonnée en de qr2",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["20"]))
print("coordonnée en de q1",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["21"]))
print("coordonnée en de q4",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["22"]))
print("coordonnée en de q3",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["23"]))"""

#print("coordonnée en de q3",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["22"]))

#draw_object(img=img,data=center)



