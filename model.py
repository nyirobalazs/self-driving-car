
from tensorflow import keras
import numpy as np
import cv2
import os

#running trained model on the car
class Model:

    saved_model = 'trained_model.h5'

    def __init__(self):
        self.model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_model))
        self.model.summary()

    def preprocess(self, image):
        image = cv2.resize(image, (240,320,3))    
        return image
    
    def predict(self, image):
        image = self.preprocess(image)
        angle, speed = self.model.predict(np.array([image]))
        angle = 80 * np.clip(angle, 0, 1) + 50
        speed = 35 * np.clip(speed, 0, 1)
        return angle, speed

