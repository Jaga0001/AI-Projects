import cv2
import imutils

img = cv2.imread('C://Users/jagaj/Desktop/Jaga/Artificial Intelligence/Day - 4/Anime.jpg')
resizedImg = imutils.resize(img, width=20) 

cv2.imwrite("Resized Image.jpg", resizedImg)