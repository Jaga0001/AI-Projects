import cv2 as cv
img = cv.imread('Logo.jpg')
cv.imshow("Anime",img)
cv.imwrite("Anime.jpg",img)
cv.waitKey(0)
cv.destroyAllWindows()
