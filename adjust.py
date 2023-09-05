import numpy as np
import cv2
import mediapipe as mp
from PIL import Image




#read image
def read_image(image_path):
    image = cv2.imread(image_path)
    return image

#show image
def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#resize image
def resize_image(image,dx):
    height, width = image.shape[:2]
    res = cv2.resize(image,(int(dx*width), int(dx*height)), interpolation = cv2.INTER_CUBIC)
    return res



#crop image in (x,y) 512x512
def crop_image(image,x,y):
    crop_img = image[y-250:y+150, x-150:x+150]
    return crop_img


def main(img):
    diffrence = 94
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        
        print(f"original image",img.shape)
        # show_image(img)
        results = face_detection.process(img)
        frame_height, frame_width, _ = img.shape
        
        if results.detections:


            face_react = np.multiply(
            [
            results.detections[0].location_data.relative_bounding_box.xmin,
            results.detections[0].location_data.relative_bounding_box.ymin,
            results.detections[0].location_data.relative_bounding_box.width,
            results.detections[0].location_data.relative_bounding_box.height,
            ],
            [frame_width, frame_height, frame_width, frame_height]).astype(int)
            # print(results.detections)
            key_points = np.array([(p.x, p.y) for p in results.detections[0].location_data.relative_keypoints])
            key_points_coords = np.multiply(key_points,[frame_width,frame_height]).astype(int)

            left_eye = key_points_coords[0]
            right_eye = key_points_coords[1]
            #diffrence between eyes
            diffrence_eyes = right_eye[0] - left_eye[0]
            # print(diffrence_eyes)
            #diffrence betwwen mouth and nose
            diffrence_mouth_nose = key_points_coords[3][1] - key_points_coords[2][1]

            resized_image=resize_image(img, diffrence/diffrence_eyes)
            result = face_detection.process(resized_image)

            frame_height, frame_width, _ = resized_image.shape

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
    
                
                # detect nose
                nose = key_points_coords[2]

                show_image(resized_image)
                print(f"resized image",resized_image.shape)
                show_image(resized_image)
                crop_imaged = crop_image(resized_image, nose[0], nose[1])
               

                print(f"cropped image",crop_imaged.shape)
                show_image(crop_imaged)
                #save crop image
                # cv2.imwrite("crop_image3.jpg", crop_image)
               
                return crop_imaged
            else:
                print("no face detected")
        else:
            print("no face detected")

img = read_image("data/far.jpg")

main(img)
