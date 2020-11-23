import cv2
import sys

img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
scale_percent = 10 


width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
cv2.imwrite("small"+sys.argv[1], resized) 
