import numpy as np
import cv2
import os

## for Face Detection
def faceDetection(input_img):

    # converting the image to bgr
    gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # specifying what we want to detect
    face_haar = cv2.CascadeClassifier(r'D:\Python\final_cse_project\haarcascade_frontalface_alt.xml')
    faces = face_haar.detectMultiScale(gray_image,scaleFactor=1.3, minNeighbors=3)
    return faces,gray_image

## preparing the training data
def labels_for_training_data(directory):
    faces=[]
    faceID = []
    names = []

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system files")
                continue

            # storing the id and name of the person
            id = os.path.basename(path).split('_')[1]
            name = os.path.basename(path)

            img_path=os.path.join(path,filename)
            print("img_path", img_path)
            print("id", id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Not Loaded properly")
                continue
            
            # for making the rectangle over the face 
            faces_rect,gray_img=faceDetection(test_img)
            if len(faces_rect)!=1:
                continue
            (x,y,w,h)=faces_rect[0]
            
            # here we crop the image to include only the face i.e. region of interest
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
            names.append(name)
    
    return faces,faceID,names
            

## training the classifier
def train_Classifier(faces, faceID):
    print(faceID)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer


## drawing the rectangle on the face
def draw_rect(test_img, face):
    (x,y,w,h) = face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=3)

## putting the name of the person
def put_text(test_img, label_name,x,y):
    cv2.putText(test_img, label_name,(x,y), cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),3)


