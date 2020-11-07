import cv2
from cv2 import dnn_superres
from stack_images import stackImages



##takes a really long time to run

sr = dnn_superres.DnnSuperResImpl_create()

image = cv2.imread('Ressources/picture_png_1.PNG')
image2 = cv2.imread('Ressources/picture_png_2.PNG')
image3 = cv2.imread('Ressources/picture_png_3.PNG')
image5 = cv2.imread('Ressources/picture_png_5.PNG')


path = "Ressources/EDSR_x4_best_performing.pb"
sr.readModel(path)

sr.setModel("edsr", 4)


result = sr.upsample(image)
result2 = sr.upsample(image2)
result3 = sr.upsample(image3)
result5 = sr.upsample(image5)

cv2.imwrite("./upscaled1.png", result)
cv2.imwrite("./upscaled2.png", result2)
cv2.imwrite("./upscaled3.png", result3)
cv2.imwrite("./upscaled5.png", result5)

imgStack = stackImages(0.5,([image,result],[image2,result2],[image3,result3],[image5,result5]))
cv2.imshow("stacked", imgStack)

cv2.waitKey(0)