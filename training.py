import numpy as np
import cv2
import os

import face_recognition as fr


test_img= cv2.imread(r'D:\Python\final_cse_project\petla.jpeg')

faces_detected, gray_image = fr.faceDetection(test_img)

#Training starts

faces,faceID,names=fr.labels_for_training_data(r'D:\Python\final_cse_project\Dataset')
face_recognizer = fr.train_Classifier(faces,faceID)
face_recognizer.save(r'D:\Python\final_cse_project\trainingData.yml')

# names = {
#     1925: "Himraj Gogoi",
#     1924: "Harsh Bordhan Singh",
#     1952: "Parikshit Borah"
# }

# for face in faces_detected:
#     (x,y,w,h) = face
#     roi_gray = gray_image[y:y+h,x:x+h]
#     label,confidence = face_recognizer.predict(roi_gray)
#     print("Confidence :",confidence)
#     print("Label: ", label)
#     fr.draw_rect(test_img,face)
#     predicted_name= names[label]
#     fr.put_text(test_img,predicted_name,x,y)


# resized_img = cv2.resize(test_img,(1000,700))

# cv2.imshow("face detection ", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

