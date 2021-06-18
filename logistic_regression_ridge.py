import numpy as np
import pickle
from preprocess import load_data
from utils import *


class PackedLogisticRegressionRidge:

    def __init__(self, X_train, y_train, X_test, y_test, categories, print_nums=20):
        self.model_type = "ridge"
        self.print_nums = print_nums
        self.n_train_samples, self.n_features = X_train.shape
        self.categories = categories
        self.y_in_each_model = self.init_y(y_train)

        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test

        self.models = [LogisticRegressionRidge(i, X_train.shape[1]) for i in
                       range(self.categories)]

        self.loss_list = []
        self.acc_list = []

    def init_y(self, y):
        y_in_each_model = [np.zeros((self.n_train_samples, 1)) for i in range(self.categories)]
        for i in range(self.n_train_samples):
            cat = int(y[i])
            # print(f"category: {cat}")
            y_in_each_model[cat][i] = 1
        return y_in_each_model

    def save_models(self):
        data = []
        path = f"./checkpoint/{self.model_type}.pth"
        for i in range(self.categories):
            data.append({
                "w": self.models[i].w,
            })
        pickle.dump(data, open(path, 'wb'))

    def load_models(self):
        path = f"./checkpoint/{self.model_type}.pth"
        data = pickle.load(open(path, 'rb'))
        for i in range(self.categories):
            self.models[i].w = data[i]['w']

    def run(self, epochs, batch_size=10000, lr=0.1, is_save=True):
        print("Start training...")
        for i in range(epochs):
            loss = self.train(epochs, batch_size, lr)
            acc = self.predict()
            if i % (epochs / self.print_nums) == 0:
                print(f"Epoch {i}, loss: {loss}, acc: {acc}")
            self.loss_list.append(loss)
            self.acc_list.append(acc)
        if is_save:
            self.save_loss_acc()
        return

    def train(self, epochs=2000, batch_size=10000, lr=0.1):
        n_train_samples = self.X_train.shape[0]
        batch_nums = int(n_train_samples / batch_size)
        total_loss = 0
        for j in range(batch_nums):
            left_bound = j * batch_size
            right_bound = n_train_samples if j == batch_nums - 1 else (j + 1) * batch_size
            loss = 0
            for k in range(self.categories):
                loss += self.models[k].train(self.X_train[left_bound:right_bound, :],
                                             self.y_in_each_model[k][left_bound:right_bound, :],
                                             lr=lr, lambda_penalty=0.5)
            loss /= self.categories
            total_loss += loss
        total_loss /= batch_nums
        return total_loss

    def predict(self):
        labels = []
        for i in range(self.categories):
            labels.append(self.models[i].predict(self.X_test))
        labels = np.array(labels).T
        labels = np.squeeze(labels)
        labels = np.argmax(labels, axis=1).reshape(-1, 1)  # (n_samples, 1)
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


class LogisticRegressionRidge:

    def __init__(self, model_idx, n_features):
        self.model_idx = model_idx
        self.n_features = n_features

        self.w = np.zeros((n_features + 1, 1))

    def train(self, X, y, lr=0.1, lambda_penalty=0.5):
        X = np.insert(X, 0, 1, axis=1)

        n_samples = X_train.shape[0]
        y_hat = sigmoid(np.dot(X, self.w))

        # print(f"Epoch: {i}, loss: {float(loss)}")
        dw = (1.0 / n_samples) * (np.dot(X.T, (y_hat - y)) + lambda_penalty * self.w)
        self.w -= lr * dw

        loss = (1 / n_samples) * (np.sum(
            np.power(y - y_hat, 2)) + np.sum(lambda_penalty * np.power(self.w, 2)))  # MSE loss with penalty
        return loss

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_hat = sigmoid(np.dot(X, self.w))  # (n_samples, 1)
        # y_label = [1 if y_hat[i] > 0.5 else 0 for i in range(X.shape[0])]
        # label = np.array(y_label)
        label = np.array(y_hat)
        return label


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(path="./data/preprocessed_data_1764.pkl", reload=True)
    print(X_train.shape)
    packed_model = PackedLogisticRegressionRidge(X_train, y_train, X_test, y_test, categories=10, print_nums=100)
    # packed_model.load_models()
    packed_model.run(epochs=300, lr=0.01)

    print(f"Test accuracy: {packed_model.acc_list[-1]}")
