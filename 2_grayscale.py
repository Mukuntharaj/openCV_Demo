import cv2

img = cv2.imread("D:/OpenCV_Freecodecamp/Dimlight/download.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()