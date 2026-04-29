import cv2

img = cv2.imread("D:/OpenCV_Freecodecamp/Dimlight/download.jpg")

resized = cv2.resize(img, (640,480))
cropped = img[50:200,100:300]
flipped = cv2.flip(img,1)

cv2.imshow("Original", img)
cv2.imshow("Resized", resized)
cv2.imshow("Cropped", cropped)
cv2.imshow("Flipped", flipped)

cv2.waitKey(0)
cv2.destroyAllWindows()