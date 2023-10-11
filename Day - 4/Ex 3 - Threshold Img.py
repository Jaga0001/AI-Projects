import cv2 
img = cv2.imread("C://Users/jagaj/Desktop/Jaga/Artificial Intelligence/Day - 4/Anime.jpg")
grayImg =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshImg = cv2.threshold(grayImg, 120, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("Threshold.jpg", threshImg)