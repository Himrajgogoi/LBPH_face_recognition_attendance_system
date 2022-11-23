import cv2
import sys
import os

cpt = 0

vidStream = cv2.VideoCapture(0)

name = input("Enter your name: ")
rollno = input("Enter the first 2 and last 2 or 3 digits of your roll number: ")

name = name.lower() 
name = name + "_" + rollno

parent_directory = r"D:\Python\final_cse_project\Dataset"
path = os.path.join(parent_directory,name)

## checking if the student folder already exists or not
if os.path.exists(path) == False:
    os.mkdir(path)

while True:

    ret, frame = vidStream.read()
    cv2.imshow('test window', frame)
    
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r"D:\Python\final_cse_project\Dataset\%s\image%04i.jpg"%(name,cpt), gray_image)
    cpt +=1
    
    if cpt == 1:
        print("give expressions")

    if cv2.waitKey(100) == ord('q') or cpt == 200:
        break