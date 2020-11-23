import cv2
import sys
#read image
img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)

#image scale in 0-100
scale_percent = 10 

#determine size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#write
cv2.imwrite("small"+sys.argv[1], resized) 
