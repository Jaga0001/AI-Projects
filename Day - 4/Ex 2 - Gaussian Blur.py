import cv2 
img = cv2.imread("C://Users/jagaj/Desktop/Jaga/Artificial Intelligence/Day - 4/Anime.jpg")
gaussianBlurImg = cv2.GaussianBlur(img, (21, 21), 0)
cv2.imwrite("Gaussian Blur Image.jpg", gaussianBlurImg)