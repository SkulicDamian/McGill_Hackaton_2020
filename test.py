import cv2
from cv2 import dnn_superres
import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


##model 1
sr = dnn_superres.DnnSuperResImpl_create()

image = cv2.imread('Ressources/picture_png_1.PNG')
image2 = cv2.imread('Ressources/picture_png_2.PNG')

path = "Ressources/ESPCN_x4_small_model.pb"
sr.readModel(path)

sr.setModel("espcn", 4)

result = sr.upsample(image)
result2 = sr.upsample(image2)

cv2.imwrite("./upscaled.png", result)

imgStack = stackImages(2.0,([image,result],[image2,result2]))
cv2.imshow("stacked", imgStack)


##model 2

sr2 = dnn_superres.DnnSuperResImpl_create()

image = cv2.imread('Ressources/picture_png_3.PNG')
image2 = cv2.imread('Ressources/picture_png_4.PNG')

path = "Ressources/EDSR_x4_best_performing.pb"
sr2.readModel(path)

sr2.setModel("edsr", 2)

result = sr2.upsample(image)
result2 = sr2.upsample(image2)

cv2.imwrite("./upscaled.png", result)

imgStack = stackImages(1.0,([image,result],[image2,result2]))
cv2.imshow("stacked", imgStack)







cv2.waitKey(0)