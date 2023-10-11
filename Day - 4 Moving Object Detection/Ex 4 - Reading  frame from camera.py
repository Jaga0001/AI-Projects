import cv2
vs =  cv2.VideoCapture(0)
while True:
    _, img= vs.read()
    cv2.imshow("cameraFeed", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
vs.release()
cv2.destroyAllWindows()