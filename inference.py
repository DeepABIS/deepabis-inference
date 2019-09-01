import keras
import cv2
import numpy as np
import json
from keras.utils import CustomObjectScope
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

path = './'


def relu6(x):
    return keras.backend.relu(x, max_value=6)


class BeeNet:
    def __init__(self):
        with CustomObjectScope({'relu6': relu6,
                                'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
            self.model = keras.models.load_model(path + 'models/beenet_17.hdf5')
            self.model._make_predict_function() # have to initialize before threading
            self.scaler = joblib.load(path + 'transform/17.pkl')
            with open(path + 'embedding/14.json') as file:
                self.embedding = json.load(file)

    def transform(self, image):
        img = np.float32(image)
        img = self.scaler.transform(img)
        img = np.reshape(img, (256, 256, 1))
        return img

    def infer(self, file):
        img = cv2.imread(file, 0)
        img = cv2.resize(img, (256, 256))
        img = self.transform(img)

        batch = np.array([img])
        predict = self.model.predict(batch)
        return predict

    def infer_top5(self, file):
        result = self.infer(file)[0]
        arg_sorted = np.flip(np.argsort(result), axis=0)
        val_sorted = np.flip(np.sort(result), axis=0)
        stacked = np.dstack((arg_sorted, val_sorted))
        squeezed = np.squeeze(stacked)
        top5 = squeezed[:5]
        return top5
