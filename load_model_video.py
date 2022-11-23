import numpy as np
import cv2
import os
import xlwt
from datetime import datetime
import time
from xlwt import Workbook

import face_recognition as fr


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'D:\Python\final_cse_project\trainingData.yml')

## For Video 
vid = cv2.VideoCapture(0)
captureTime = 5

size= (
    int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

## student database
names = {
    1925: "Himraj Gogoi",
    1924: "Harsh Bordhan Singh",
    1952: "Parikshit Borah",
    2003: "Anuj  Barman"
}

##attendance sheet
wb = Workbook()

title = "Attendance on " + datetime.today().strftime('%Y-%m-%d')
sheet1 = wb.add_sheet(title, cell_overwrite_ok=True)

sheet1.write(0,0, 'Roll Number')
sheet1.write(0,1, 'Name')

startTime = time.time()
while (int(time.time()-startTime) < captureTime):

    res,test_img = vid.read()

    faces_detected, gray_image = fr.faceDetection(test_img)
    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_image[y:y+h,x:x+h]
        label,confidence = face_recognizer.predict(roi_gray)
        print("Confidence :",confidence)
        print("Label: ", label)
        fr.draw_rect(test_img,face)
        predicted_name= names[label]
        if(confidence > 70):
            fr.put_text(test_img,"unknown",x,y)
            continue
        fr.put_text(test_img,predicted_name,x,y)
        sheet1.write(1,0,str(label))
        sheet1.write(1,1,names[label])

    resized_img = cv2.resize(test_img,(1000,700))

    cv2.imshow("face detection ", resized_img)

    if cv2.waitKey(10) == ord('q'):
        break
wb.save('attendance.xls')





