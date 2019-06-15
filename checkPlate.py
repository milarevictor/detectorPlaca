
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from skimage.filters import threshold_otsu


from imutils import contours
from imutils.perspective import four_point_transform

cam = cv2.VideoCapture(0)
ret, frame = cam.read()
img_name = "imagePlate.png"
cv2.imwrite(img_name, frame)
cam.release()
cv2.destroyAllWindows()
img_path = 'imagePlate.png'
img_new    = cv2.imread(img_path,0)
img_clean = cv2.imread(img_path)
temp = cv2.imread(img_path)
img = cv2.blur(img_new,(6,6))
blur = cv2.blur(img_new,(2,2)).astype('uint8')
gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plt.imshow(th2, cmap="gray")
# plt.show()

laplacian   = cv2.Laplacian(img,cv2.CV_64F)
sobelx      = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
sobely      = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)


tmp, imgThs = cv2.threshold(sobelx,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)

thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
krl      = np.ones((6,6),np.uint8)
erosion  = cv2.erode(thresh,krl,iterations = 1)
krl      = np.ones((12,4),np.uint8)
dilation = cv2.dilate(erosion,krl,iterations = 1)
thresh   = dilation
plate_like_objects = []
morph = cv2.getStructuringElement(cv2.MORPH_RECT,(45,30))

plateDetect = cv2.morphologyEx(th2,cv2.MORPH_CLOSE,morph)
regionPlate = plateDetect.copy()

cnts = cv2.findContours(regionPlate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

cnts = cnts[0] if imutils.is_cv2() else cnts[1]
rects = []

for c in cnts:
        epsilon = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.03*epsilon,True)
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        if w>=200 and w<=800 and h>=100 and ar>=2.8 and ar<=4:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rects.append(box)
            cv2.drawContours(temp,[box],0,(0,0,255),2)
if rects:
    for rect in rects:
        s = rect.sum(axis=1)
        top_left = rect[np.argmin(s)]
        bottom_right = rect[np.argmax(s)]
        x_1 = int(top_left[0])
        y_1 = int(top_left[1])
        x_2 = int(bottom_right[0])
        y_2 = int(bottom_right[1])
        try:
            # plt.imshow(img_clean[y_1:y_2, x_1:x_2], cmap="gray")
            # plt.show()

            plate_like_objects.append(th2[y_1:y_2, x_1:x_2])
        except:
            print("")
