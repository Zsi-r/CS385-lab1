import numpy as np
import pickle
from preprocess import load_data
from utils import *
from sklearn.svm import SVC, LinearSVC

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(path="../data/preprocessed_data.pkl", reload=True)
    print(X_train.shape)
    print("Training...")
    model = SVC(C=0.5, random_state=42)
    batch_size = 10000
    batch_num = int(X_train.shape[0] / batch_size)
    for i in range(batch_num):
        print(f"Epoch {i}...")
        right_bound = (i+1) * batch_size
        if i == batch_num - 1:
            right_bound = max(right_bound, X_train.shape[0])
        model.fit(X_train[i * batch_size:right_bound, :], np.squeeze(y_train[i*batch_size:right_bound, :]))

    print("Predicting...")
    acc = model.score(X_test, y_test)
    print(f"test accuracy: {acc}")
