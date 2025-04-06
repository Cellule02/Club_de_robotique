import cv2
import cv2.aruco as aruco
import time
import numpy as np 

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detectorParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detectorParams)

real_xy_QRcode = {
    "21":(600,1400),
    "22":(2400,1400),
    "23":(600,600),
    "24":(2400,600)
}

def get_center(pos):
    pos = pos[0]
    x = [float(i[0]) for i in pos]
    y = [float(i[1]) for i in pos]

    center_x = min(x) + (max(x)-min(x))/2
    center_y = min(y) + (max(y)-min(y))/2
    #print(center_x,center_y)
    return center_x, center_y

def get_aruco_id(img, detect=detector):
    try:
        data_center = {}
        data = {}
        marker_corners, marker_ids, rejected_candidates = detect.detectMarkers(img)
        ids = marker_ids.transpose()[0]
        for i in range(len(marker_ids)):
            data_center[ids[i]] = get_center(marker_corners[i])
            data[ids[i]] = marker_corners[i]
        return data,data_center
    except AttributeError:
        pass


def draw_object(img,data):
    try:
        for key,value in data.items():
            value = value[0]
            #print(key,"value is ", value[0],value[1])
            cv2.line(img, [int(i) for i in value[0]],[int(i) for i in value[1]],(255,0,0),thickness=2)
            cv2.line(img, [int(i) for i in value[0]],[int(i) for i in value[3]],(255,0,0),thickness=2)
            cv2.line(img, [int(i) for i in value[1]],[int(i) for i in value[2]],(255,0,0),thickness=2)
            cv2.line(img, [int(i) for i in value[2]],[int(i) for i in value[3]],(255,0,0),thickness=2)

            cv2.putText(img,str(key),[int(i)-10 for i in value[2]],cv2.FONT_HERSHEY_COMPLEX,1,color=(0,255,0),thickness=2)
    except:
        pass

    cv2.imshow("test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



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


initial_corners = {
            "23HG": np.array([60, 60]),
            "22HD": np.array([240, 60]),
            "21BG": np.array([140, 60]),
            "20BD": np.array([240, 140])
        }
mire_size = int

mindex= 23
mire_size = 10


#x0 = lambda yobj: round(initial_corners[mires_all_co[mindex][2]][0]-pxl2m_coef*(yobj-mires_all_co[mindex][1]),3)
#y0 = lambda xobj: round(initial_corners[mires_all_co[mindex][2]][1]-pxl2m_coef*(xobj - mires_all_co[mindex][0]),3) # faux


img = cv2.imread("camera/imgs/img_test/test1_screenshot_20.02.2025.png")
data, data_center = get_aruco_id(img)

pxl2m_coef = (initial_corners["23HG"]-initial_corners["22HD"])/(data[23][0]-data[22][0])
pxl2m_coef=pxl2m_coef[0]
print(pxl2m_coef)
print(initial_corners["23HG"][0])
print(data_center[mindex][0])


x0 = lambda xobj: initial_corners["23HG"][0]-(xobj-data_center[mindex][0])*pxl2m_coef
y0 = lambda yobj: initial_corners["23HG"][1]-(yobj-data_center[mindex][1])*pxl2m_coef

print(x0(data_center[21][0]))
print(y0(data_center[21][1]))
# re_irl = re_num
# obj_irl = obj_num 

# obj_num*re_irl / re_num

draw_object(img,data)
"""cv2.imshow("t",img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""




