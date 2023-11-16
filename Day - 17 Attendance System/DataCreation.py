import imutils
import time
import cv2
import csv
import os

cascade = 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade)  # Create haar cascade

Name = str(input("Enter your name: "))
Rollno = int(input("Enter your roll number: "))
dataset = 'dataset'
sub_data = Name
path = os.path.join(dataset, sub_data)

if not os.path.isdir(path):
    os.mkdir(path)
    print(sub_data)

info = [Name, Rollno]
with open('StudentDetails.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(info)
csvFile.close()

print("[INFO] Initializing face capture. Look the camera and wait ...")
cam = cv2.VideoCapture(0)
time.sleep = 2.0
total = 0

while total < 50:
    print(total)
    _, frame = cam.read()
    img = imutils.resize(frame, width=400)
    rects = detector.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5,
                                      minSize=(30, 30))

    for x, y, w, h in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        p = os.path.sep.join([path, "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, frame)
        total += 1

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
