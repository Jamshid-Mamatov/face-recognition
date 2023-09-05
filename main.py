import mediapipe
import cv2
import numpy as np
from math import pi

def read_img(path):
    img = cv2.imread(path)
    return img

def show_img(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def rotate2D(img,x,y):
#     print(x,y)
#     x1,x2 = x
#     y1,y2 = y
#     #how define angle x1,x2,y1,y2 if 
#     angle = np.arctan((y2-y1)/(x2-x1))
#     print(angle)
#     img = cv2.circle(img,(x1,y1),15,(0,0,255),-1)
#     img = cv2.circle(img,(x2,y2),15,(0,0,255),-1)
#     return img



def main(name):
    img = read_img(name)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_detector=mediapipe.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    results = face_detector.process(rgb_img)
    frame_height, frame_width, _ = rgb_img.shape
    # print(results.detections)
    #get bounding box and landmarks
    if results.detections:
        face_react = np.multiply(
        [
        results.detections[0].location_data.relative_bounding_box.xmin,
        results.detections[0].location_data.relative_bounding_box.ymin,
        results.detections[0].location_data.relative_bounding_box.width,
        results.detections[0].location_data.relative_bounding_box.height,
        ],
        [frame_width, frame_height, frame_width, frame_height]).astype(int)

        key_points = np.array([(p.x, p.y) for p in results.detections[0].location_data.relative_keypoints])
        key_points_coords = np.multiply(key_points,[frame_width,frame_height]).astype(int)
        x,y,h,w = face_react
        cx = x + w//2
        cy = y + h//2
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
        
    show_img(img)



main("data/norm.jpg")



