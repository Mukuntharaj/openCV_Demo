import cv2

img = cv2.imread("D:/OpenCV_Freecodecamp/Dimlight/download.jpg")

blur = cv2.blur(img,(5,5))
gaussian = cv2.GaussianBlur(img,(5,5),0)

cv2.imshow("Original", img)
cv2.imshow("Average Blur", blur)
cv2.imshow("Gaussian Blur", gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()