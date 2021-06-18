import numpy as np
import pickle
from preprocess import load_data
from utils import *


class PackedLDA:

    def __init__(self, X_train, y_train, X_test, categories):
        self.model_type = "lda"
        self.n_train_samples, self.n_features = X_train.shape
        self.categories = categories
        self.y_in_each_model = self.init_y(y_train)

        self.models = [LDA(X_train, self.y_in_each_model[i], X_test, i) for i in
                       range(self.categories)]

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

    def train(self, batch_size=10000):
        print("Start training...")
        for i in range(self.categories):
            self.models[i].train(batch_size=batch_size)
        return

    def predict(self):
        print("Start predicting...")
        labels = []
        for i in range(self.categories):
            labels.append(self.models[i].predict())
        labels = np.array(labels).T
        labels = np.squeeze(labels)
        # print(labels.shape)
        labels = np.argmax(labels, axis=1).reshape(-1, 1)
        # print(labels.shape)
        return labels

    def save_loss(self):
        save_loss_list = []
        for i in range(self.categories):
            save_loss_list.append(self.models[i].loss_list)
        save_loss_list = np.array(save_loss_list)
        save_loss_list = np.mean(save_loss_list, axis=0)

        dirname = "./log"
        filename = f"{self.model_type}_" + time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(dirname, filename)
        f = open(path, 'w')
        f.write("epoch,loss\n")
        for i, loss in enumerate(save_loss_list):
            if i % 100 == 0:
                print(f"Epoch {i}, loss: {loss}")
            f.write(f"{i},{loss}\n")
        f.close()


class LDA:

    def __init__(self, X_train, y_train, X_test, model_idx):
        self.model_idx = model_idx
        self.n_train_samples, self.n_features = X_train.shape


        self.X_train = X_train
        self.X_train = np.insert(X_train, obj=0, values=1, axis=1)
        self.y_train = y_train
        self.X_test = X_test
        self.X_test = np.insert(X_test, obj=0, values=1, axis=1)
        self.loss_list = []

        # self.w = np.zeros((self.n_features, 1))
        # self.mu_pos = np.zeros((self.n_features, 1))
        # self.mu_neg = np.zeros((self.n_features, 1))
        # self.S_W = np.zeros((self.n_features, self.n_features))
        self.w = np.zeros((self.n_features + 1, 1))
        self.mu_pos = np.zeros((self.n_features + 1, 1))
        self.mu_neg = np.zeros((self.n_features + 1, 1))
        self.S_W = np.zeros((self.n_features + 1, self.n_features + 1))

    def calc_s_mu(self, X_train, y_train):
        # if self.model_idx == 0:
        #     print(X_train.shape)
        y_squeeze = np.squeeze(y_train)
        # print(y_squeeze)
        X_one = X_train[y_squeeze == 1, :]
        X_zero = X_train[y_squeeze == 0, :]
        # print(f"X_one: {X_one.shape}")
        self.mu_pos = np.mean(np.squeeze(X_one), axis=0)[:, np.newaxis]  # (n_feature, 1)
        self.mu_neg = np.mean(np.squeeze(X_zero), axis=0)[:, np.newaxis]
        # print(f"mu_pos: {self.mu_pos.shape}")
        sigma_pos = np.cov(X_one.T)
        # print(f"X_zero: {sigma_pos.shape}")
        sigma_neg = np.cov(X_zero.T)
        # print(f"sigma_pos: {sigma_pos.shape}")

        n_pos = np.count_nonzero(self.y_train)
        n_neg = self.n_train_samples - n_pos
        # self.S_B = np.dot((self.mu_pos - self.mu_neg), (self.mu_pos - self.mu_neg).T)  # (n_feature, n_feature)
        self.S_W = n_pos * sigma_pos + n_neg * sigma_neg
        # print(f"S_W: {self.S_W.shape}")
        # print(f"S_B: {self.sigma_neg.shape}")

    # def calc_sigma(self, y_squeeze):
    #     sigma_pos = np.zeros((self.n_features + 1, self.n_features + 1))
    #     sigma_neg = np.zeros((self.n_features + 1, self.n_features + 1))
    #     X_one = self.X_train[y_squeeze == 1, :]
    #     X_zero = self.X_train[y_squeeze == 0, :]
    #
    #     for i, row in enumerate(X_one):
    #         sigma_pos += np.dot((row - self.mu_pos), (row - self.mu_pos).T)
    #     for i, row in enumerate(X_zero):
    #         sigma_neg += np.dot((row - self.mu_neg), (row - self.mu_neg).T)
    #     return sigma_pos, sigma_neg

    def train(self, batch_size=10000):
        batch_num = int(self.n_train_samples/batch_size)

        for i in range(batch_num):
            right_bound = (i + 1) * batch_size
            if i == batch_num - 1:
                right_bound = max(right_bound, self.n_train_samples)
            # print(f"shape: {self.y_train.shape}")
            self.calc_s_mu(self.X_train[i * batch_size: right_bound, :], self.y_train[i * batch_size: right_bound, :])
            # print(f"shape: {np.linalg.inv(self.S_W).dot((self.mu_pos - self.mu_neg))}")
            # print(f"mark: {self.S_W}")
            # print(f"mu_pos: {self.mu_pos}")
            # print(f"train {self.X_train[0]}")
            # print(self.X_train[0])
            # print(self.mu_pos[100])
            self.w += np.linalg.pinv(self.S_W).dot((self.mu_pos - self.mu_neg))
        self.w /= batch_num
        return

    def predict(self):
        X = self.X_test
        y_hat = sigmoid(np.dot(X, self.w))  # (n_samples, 1)
        # y_label = [1 if y_hat[i] > 0.5 else 0 for i in range(X.shape[0])]
        # label = np.array(y_label)
        label = np.array(y_hat)
        # print(label)
        return label


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(path="./data/preprocessed_data_1764.pkl", reload=True)

    packed_model = PackedLDA(X_train, y_train, X_test, categories=10)
    # packed_model.load_models()
    packed_model.train(batch_size=10000)
    predict_label = packed_model.predict()
    # packed_model.save_loss()
    # packed_model.save_models()

    acc = np.mean(np.equal(predict_label, y_test))
    print(f"test accuracy: {acc}")
