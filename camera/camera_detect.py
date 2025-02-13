import cv2
import cv2.aruco as aruco

# Load the image
image = cv2.imread("camera/imgs/plateau.png")

# Define the ArUco dictionary and detector parameters
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detectorParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detectorParams)

real_xy_QRcode = {
    "21":(600,1400),
    "22":(2400,1400),
    "23":(600,600),
    "24":(2400,600)
}

def get_aruco_id(img, detect=detector):
    data = {}
    marker_corners, marker_ids, rejected_candidates = detect.detectMarkers(img)
    ids = marker_ids.transpose()[0]
    for i in range(len(marker_ids)):
        data[ids[i]] = marker_corners[i]
    return data

def get_center(pos):
    pos = pos[0]
    x = [float(i[0]) for i in pos]
    y = [float(i[1]) for i in pos]

    center_x = min(x) + (max(x)-min(x))/2
    center_y = min(y) + (max(y)-min(y))/2
    #print(center_x,center_y)
    return center_x, center_y



# Process the image and draw markers
data = get_aruco_id(image)
print(data[22])
get_center(data[22])



