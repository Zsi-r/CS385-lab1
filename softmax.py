import numpy as np
import pickle
from preprocess import load_data
from utils import *


class SoftmaxRegression:

    def __init__(self, X_train, y_train, X_test, y_test, categories, print_nums=20):
        self.model_type = "softmax"
        self.print_nums = print_nums
        self.n_train_samples, self.n_features = X_train.shape
        self.categories = categories

        self.X_train = X_train
        self.y_train = np.zeros((self.n_train_samples, categories))
        for i in range(self.n_train_samples):
            self.y_train[i][int(y_train[i])] = 1
        self.X_test = X_test
        self.y_test = y_test

        self.loss_list = []
        self.acc_list = []

        self.w = np.zeros((self.n_features + 1, self.categories))

    # def save_models(self):
    #     data = []
    #     path = f"./checkpoint/{self.model_type}.pth"
    #     for i in range(self.categories):
    #         data.append({
    #             "w": self.models[i].w,
    #         })
    #     pickle.dump(data, open(path, 'wb'))

    # def load_models(self):
        # path = f"./checkpoint/{self.model_type}.pth"
        # data = pickle.load(open(path, 'rb'))
        # for i in range(self.categories):
        #     self.models[i].w = data[i]['w']

    def run(self, epochs, batch_size=10000, lr=0.1, is_save=True):
        print("Start training...")
        for i in range(epochs):
            loss = self.train_batch(epochs, batch_size, lr)
            acc = self.predict(self.X_test)
            if i % (epochs / self.print_nums) == 0:
                print(f"Epoch {i}, loss: {loss}, acc: {acc}")
            self.loss_list.append(loss)
            self.acc_list.append(acc)
        if is_save:
            self.save_loss_acc()
        return

    def train_batch(self, epochs=2000, batch_size=10000, lr=0.1):
        n_train_samples = self.X_train.shape[0]
        batch_nums = int(n_train_samples / batch_size)
        total_loss = 0
        for j in range(batch_nums):
            left_bound = j * batch_size
            right_bound = n_train_samples if j == batch_nums - 1 else (j + 1) * batch_size
            loss = self.train(self.X_train[left_bound:right_bound, :],
                                            self.y_train[left_bound:right_bound, :],
                                            lr=lr)
            total_loss += loss
        total_loss /= batch_nums
        return total_loss
    
    def train(self, X, y, lr=0.1):
        X = np.insert(X, 0, 1, axis=1)  # (samples, n_features + 1)

        n_samples = X_train.shape[0]
        y_hat = softmax(np.dot(X, self.w)) # (samples, categories)

        # print(f"Epoch: {i}, loss: {float(loss)}")
        dw = (1.0 / n_samples) * np.dot(X.T, (y_hat - y))  # (n_features+1, categories)
        self.w -= lr * dw

        loss = (1 / n_samples) * np.sum(
            np.power(y - y_hat, 2))  # MSE loss
        return loss

    def predict(self, X):
        labels = []

        X = np.insert(X, 0, 1, axis=1)
        labels = np.argmax(softmax(np.dot(X, self.w)), axis=1)  # (n_samples, 1)
        acc = np.mean(np.equal(labels, self.y_test))
        return acc

    def save_loss_acc(self):
        loss_acc = zip(self.loss_list, self.acc_list)
        dirname = "./log"
        filename = f"{self.model_type}" + time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(dirname, filename)
        f = open(path, 'w')
        f.write("epoch,loss,acc\n")
        for i, value in enumerate(loss_acc):
            f.write(f"{i},{value[0]},{value[1]}\n")
        f.close()


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(path="./data/preprocessed_data_1764.pkl", reload=True)
    print(X_train.shape)
    packed_model = SoftmaxRegression(X_train, y_train, X_test, y_test, categories=10, print_nums=100)
    # packed_model.load_models()
    packed_model.run(epochs=300, lr=0.1)

    print(f"Test accuracy: {packed_model.acc_list[-1]}")
