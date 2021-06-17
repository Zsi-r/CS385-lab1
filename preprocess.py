from PIL import Image
from scipy.io import loadmat
import numpy as np
from skimage.feature import hog
import pickle


def load_data(path="../data/preprocessed_data.pkl", reload=True):
    if reload:
        data = pickle.load(open(path, 'rb'))
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
    else:
        X_train, y_train = preprocess("../data/train_32x32.mat")
        X_test, y_test = preprocess("../data/test_32x32.mat")
        pickle.dump({"X_train": X_train,
                     "y_train": y_train,
                     "X_test": X_test,
                     "y_test": y_test}, open(path, 'wb'))
    return X_train, y_train, X_test, y_test


def preprocess(path="./data/train_32x32.mat"):
    # load data
    f = loadmat(path)
    X = f['X']
    y = f['y']

    # convert RGB image to gray image
    X_res = []
    for i in range(X.shape[-1]):
        feature = hog(X[..., i], orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2))  # 1764 features
        # feature = hog(X[..., i])  # 324 features
        X_res.append(feature)
    X_res = np.array(X_res)

    # replace label 10 for label 0
    y[y == 10] = 0

    return X_res, y
