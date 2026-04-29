import cv2

img = cv2.imread("D:/OpenCV_Freecodecamp/Dimlight/download.jpg")

if img is None:
    print("Image not found")
else:
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()