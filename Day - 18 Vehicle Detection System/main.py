import cv2
import imutils

# Load the cascade
car_cascade = cv2.CascadeClassifier('cars.xml')
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    b = str(len(cars))
    a = int(b)
    detected = a
    n = detected
    print("------------------")
    print("Detected Vehicles: ", n)
    if n >= 2:
        print("Status: Traffic Jam")
    else:
        print("Status: No Traffic Jam")
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
