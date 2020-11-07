import cv2
from cv2 import dnn_superres


sr = dnn_superres.DnnSuperResImpl_create()

image = cv2.imread('Ressources/picture_png_1.PNG')

path = "Ressources/ESPCN_x4_small_model.pb"
sr.readModel(path)

sr.setModel("espcn", 3)

result = sr.upsample(image)

cv2.imwrite("./upscaled.png", result)


cv2.imshow("Original Image", image)
cv2.imshow("upscaled image", result)