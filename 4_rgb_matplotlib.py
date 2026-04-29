import cv2
import matplotlib.pyplot as plt

img = cv2.imread("D:/OpenCV_Freecodecamp/Dimlight/download.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(rgb)
plt.axis("off")
plt.show()