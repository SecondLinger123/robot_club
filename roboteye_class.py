import cv2

class roboteye:

    #initialize thinfs
    def __init__(self):
       pass
    
    #get image and return array
    def return_image(string):
       image = cv2.imread(string)
       return image
