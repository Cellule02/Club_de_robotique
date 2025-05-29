import cv2 as cv
import cv2.aruco as aruco
import time
import numpy as np 
import matplotlib.pyplot as plt 
import glob
import socket
from vect import corrigeDeformation, get_theta, rotate_z, rotate_x, f_dist

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detectorParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detectorParams)

"""MTX = np.array([[466.55934433, 0, 327.07744536],
    [0, 465.74527473,252.85468805],
    [0, 0, 1]])
DIST = np.array([[-0.45682361, 0.31626673, 0.00050067, -0.0042928, -0.14752272]])
RVECS=(np.array([[-0.05830831],
       [ 0.73637161],
       [ 3.03878495]]), np.array([[-0.06110835],
       [ 0.7380924 ],
       [ 3.03836588]]), np.array([[-0.06232498],
       [ 0.72339704],
       [ 3.01244847]]), np.array([[-0.46674298],
       [-0.04301796],
       [-0.05394682]]), np.array([[-4.68761857e-01],
       [-3.07705484e-02],
       [-6.74902523e-05]]))"""
CALSTART = False
COLOR="blue"


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
    #print(len(images))
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

def save_param(file_path):
    mtx, dist, rvecs,tvecs = calcam(file_path)
    with open("camera/spec.txt", "+w") as file:
        data={'mtx':mtx, 'dist':dist, 'rvecs':rvecs, 'tvecs':tvecs}
        file.write(str(data))

def load_param():
    with open("camera/spec.txt", "r") as file:
        spec = file.read()
        spec = spec.replace('array', 'np.array')
        spec  =  eval(spec, {"np": np})
        #print(spec.keys())
    return spec["mtx"], spec["dist"], spec["rvecs"], spec["tvecs"]

def get_center(pos):
    #print(pos)
    pos = pos[0]
    
    x = [float(i[0]) for i in pos]
    y = [float(i[1]) for i in pos]

    center_x = int((max(x)+min(x))/2)
    center_y = int((max(y)+min(y))/2)
    #print(center_x,center_y)
    return np.array([center_x, center_y])

def eq_xy(qr_center):
    x22_x20 = int((qr_center[2][0] + qr_center[0][0])/2)
    x23_x21 = int((qr_center[3][0] + qr_center[1][0])/2)

    y21_y20 = int((qr_center[1][1]+qr_center[0][1])/2)
    y23_y22 = int((qr_center[3][1]+qr_center[2][1])/2)

    mean_align_center={
        "20":(x22_x20,y21_y20),
        "21":(x23_x21,y21_y20),
        "22":(x22_x20,y23_y22),
        "23":(x23_x21,y23_y22)
    }
    return mean_align_center

def get_aruco_id(img, detect=detector):
    try:
        data = {}
        marker_corners, marker_ids, rejected_candidates = detect.detectMarkers(img)
        ids = marker_ids.transpose()[0]
        for i in range(len(marker_ids)):
            #print("marker corner i",marker_corners[i])
            data[str(ids[i])] = marker_corners[i]
        return data
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
            #print("trouvé")
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
    #show_img(contours)
    #print("gradin",gradins)
    return gradins, gradins_center

def detect_gradinV2(img):
    gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edge = cv.Canny(gray, 50,200)
    contours =cv.findContours(edge,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)[0]

    #Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
    cntrRect = []
    for i in contours:
            epsilon = 0.05*cv.arcLength(i,True)
            approx = cv.approxPolyDP(i,epsilon,True)
            x,y,w,h = cv.boundingRect(i)
            if (len(approx) == 4): 
                if ((h/3 > w) or (w/3 > h)) and (h*w > 200) and (x>0) and (y>0):
                    cv.drawContours(img,approx,-1,(0,255,0),2)
                    cntrRect.append(approx)
    return cntrRect, img

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
            #print("erreur blue")
            return 0
        
def get_yellowbot(acuro_pos):
    for key in acuro_pos.keys():
        key=int(key)
        if (key >=71) and (key<=90):
            return acuro_pos[str(key)]
        else:
            #print("erreur jaune")
            return 0 

def get_vendengeuse(color, pos):
    bots = [get_bluebot(pos), get_yellowbot(pos)]
    if (color=="blue"):
        return bots
    else:
        return bots[:,:,-1]

def img_undisort(img, mtx,dist):
    
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    return dst

def points_undisort(points, mtx,dist):
    undistorted_points = cv.undistortPoints(
    points, 
    mtx, 
    dist, 
    None, 
    mtx  # Utiliser la même matrice de caméra pour la projection
    )
    undistordp = np.array([coor[0] for coor in undistorted_points])
    return undistordp

def dict2array(dict):
    sort_dict = sorted(dict.keys())
    center_list = []
    for key in sort_dict:
        center_list.append(dict[key])
    center_list=np.array(center_list, dtype=np.float32)
    return center_list

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

def send_data(addrip,data):
    import socket
    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip = addrip
    port = 5005
    serverAddress = (ip, port)
    socket.connect(serverAddress)
    message = data.encode()
    socket.send(message)

#print("notre robot ",get_vendengeuse("blue",center))
#img_obgj_detect=draw_object(img, gradins)
#print(corrigeDeformation(center["21"],center["20"],center["23"],center["22"],gradins_center[0]))
#img_obgj_detect2=draw_object(img_obgj_detect, data)
#show_img(img_obgj_detect)
# Process the image and draw markers


#mtx, dist = calcam("camera/cprb1/*.jpg")


#img = cv.imread('camera/imgs/test2/test_screenshot_16.04.20250.png')
"""img = cv.imread("camera/imgs/img_test/test_screenshot_20.02.2025.png", 1)

dst=img_undisort(img, mtx=MTX, dist=DIST)

img = cv.imread("camera/imgs/img_test/test_screenshot_20.02.2025.png", 1)
"""

#draw_object(img,data)
url = "/dev/video0"

cap = cv.VideoCapture(url)


while True:
    # recuperer l'images
    #start = time.time()
    #ret, frame = cap.read()
    ret=1
    frame = cv.imread("camera/imgs/test2/test_screenshot_16.04.20250.png", 1)
    fheight,fwidth, fchannel=frame.shape
    if CALSTART==True:
        save_param("camera/cprb2 /*.jpg")

    if ret == True:
        MTX, DIST, RVECS, _ = load_param()
        dst=img_undisort(frame, mtx=MTX, dist=DIST)
        undi_frame=reverse_perspective(dst,MTX,RVECS[0])
        #show_img(undi_frame)

        #detecter les arucos id
        arucodata = get_aruco_id(undi_frame)
        #print(arucodata)
        # on corrige les coordonnées est on trouve les centres
        """aruco_undistorted=arucodata
        for idx in arucodata.keys():
            aruco_undistorted[idx]=[points_undisort(i,mtx=MTX,dist=DIST) for i in arucodata[idx][0]]"""
        arucocenter={}
        for idx in arucodata.keys():
            arucocenter[idx]=get_center(arucodata[idx])

        # detecter notre robot
        ally, enemi = get_vendengeuse(COLOR,arucocenter)
        ally=arucodata["22"]
        enemi = arucodata["21"]
        true_ally_corner = []
        for acorner in ally[0]:
            true_ally_corner.append(corrigeDeformation(arucocenter["21"],arucocenter["20"],arucocenter["23"],arucocenter["22"],acorner))

        true_enemy_corner= []
        for ecorner in enemi[0]:
            true_enemy_corner.append(corrigeDeformation(arucocenter["21"],arucocenter["20"],arucocenter["23"],arucocenter["22"],ecorner))
            
        #print(true_ally_corner)
        #print(true_enemy_corner)
        true_ally_center = get_center([true_ally_corner])
        true_enemy_center = get_center([true_enemy_corner])
        print("fisrt corner", true_ally_corner[0])
        gradin_orientation = [get_theta(true_ally_center, gradin_center) for gradin_center in true_gradin_center]

        if is_colision(true_ally_center,true_enemy_center,100) ==False:
        

            #detecter les gradins
            gradins, imgdraw = detect_gradinV2(undi_frame)
            #print(len(gradins))
            #print(gradins[0][0][0])
            cv.circle(undi_frame,gradins[0][0][0], 1,(255,255,255), 10)
            cv.circle(undi_frame,gradins[0][1][0], 1,(255,255,255), 10)
            cv.circle(undi_frame,gradins[0][2][0], 1,(255,255,255), 10)
            cv.circle(undi_frame,gradins[0][3][0], 1,(255,255,255), 10)

            #show_img(undi_frame)
            true_gradin_corner= []
            true_gradin_center=[]
            for gradin in gradins:
                t = []
                for corner in gradin[:,0]:
                    t.append(corrigeDeformation(arucocenter["21"],arucocenter["20"],arucocenter["23"],arucocenter["22"],corner))
                true_gradin_center.append(get_center([t]))
                true_gradin_corner.append(t)
            
            
            
            
            
            data = {
                "ally_center": true_ally_center,
                "enemy_center": true_enemy_center,
                "gradins_center": true_gradin_center,
                "ally_corner": true_ally_corner,
                "enemy_corner": true_enemy_corner,
                "gradins_corner": true_gradin_corner,
                "gradin_orientation": gradin_orientation,
                "collision":False,

            }
            send_data("0.0.0.0", data)
        else:
            data = {
                "ally_center": true_ally_center,
                "enemy_center": true_enemy_center,
                "gradins_center": None,
                "ally_corner": true_ally_corner,
                "enemy_corner": true_enemy_corner,
                "gradins_corner": None,
                "gradin_orientation": None,
                "collision":True,

            }
            send_data("0.0.0.0", data)

        print(data)
        #print(time.time()-start)
        #break
    else:
        print("pas d'img")
        break



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


#corrigeDeformation(QR1_Virt,QR2_Virt,QR3_Virt,QR4_Virt,QR_Robot_Virt)

"""print("coordonnée en de qr2",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["20"]))
print("coordonnée en de q1",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["21"]))
print("coordonnée en de q4",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["22"]))
print("coordonnée en de q3",corrigeDeformation(center["21"],center["20"],center["23"],center["22"],center["23"]))
"""