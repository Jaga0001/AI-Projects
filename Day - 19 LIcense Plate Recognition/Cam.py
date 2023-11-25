import cv2
import time
import json
import base64
import requests

from auth_key import SECRET_KEY

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        print("Capturing image...")
        cv2.imwrite('test.jpg', frame)
        with open('test.jpg', 'rb') as imgfile:
            img_base64 = base64.b64encode(imgfile.read())
        url = 'https://api.openalpr.com/v2/recognize_bytes?recognize_vehicle=1&country=us&secret_key=%s' % (SECRET_KEY)
        r = requests.post(url, data=img_base64)
        num_plate = json.dumps(r.json(), indent=2)
        info = (list(num_plate.split("candidates")))
        plate = info[1]
        plate = plate.split(',')[0:3]
        p = plate[0]
        p1 = p.split(":")
        number = p1[2]
        number = number.replace('"', ' ')
        number = number.lstrip()
        print(number)

    elif key == 27:
        break
