import cv2 as cv
img = cv.imread('Logo.jpg')
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Anime",img)
cv.imshow("GrayImg",grayImg)
cv.imwrite("Gray Anime.jpg",grayImg)
cv.waitKey(0)
cv.destroyAllWindows()