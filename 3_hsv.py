import cv2

img = cv2.imread("D:/OpenCV_Freecodecamp/Dimlight/download.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("Original", img)
cv2.imshow("HSV Image", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()