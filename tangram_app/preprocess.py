from PIL import Image
import cv2
import numpy as np

# PREPROCESS FUNCTION FOR PREDICTION ON A SINGLE FRAME
def preprocess_image(img, size=250, side='left'): 
       '''
    this function takes a cv image as input, calls the resize function, chooses the left / right half of the board 
    
    Parameters : 
    img = OpenCV image
    side = process either left/right side - left by default
    size = Resize image to expected size for the model 
    '''

    if side == "left" :
        img = img[:img.shape[1]//2, :img.shape[1]//2]
    elif side == "right":
        img = img[:img.shape[1]//2, img.shape[1]//2 :]
    else:
        pass
 
    #Resize image to expected size for the model and expansion of dimension from 3 to 4
    img = cv2.resize(img, (size,size))
    # img_test = img.copy()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #no need no more cause we used the 
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    # img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))" we use this if the expands function doesn't work which it does sometimes "
    return img



