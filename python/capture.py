import cv2
import numpy as np
class capture(cv2.VideoCapture):
    def __init__(self, file_name = 0):
        super.__init__(file_name)
    
    def fr(self):
        ret, frame = self.read()
        assert frame is not None, "No webcam detected"
        processed_frame = 

    def pre_process(self):
        img = cv2.imread('bus.jpg')
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img,(640,640))/255.

        img2 = np.transpose(img2, axes=[2,0,1])
        print(img2.shape)
        img2 = np.reshape(img2, (1,3,640,640))