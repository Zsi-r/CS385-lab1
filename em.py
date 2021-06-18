import numpy as np
import pickle
from preprocess import load_data
from utils import *
from scipy.stats import multivariate_normal


class EM:

    def __init__(self, X_train, y_train, X_test, categories):
        self.model_type = "em"
        self.n_train_samples, self.n_features = X_train.shape
        self.categories = categories
        self.loss_list = []

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

        self.w = (1.0 / self.categories) * \
                 np.ones((self.n_train_samples, self.categories))  # (n_train_samples, categories) 隐变量
        self.mu, self.var = self.calc_mu_var()  # (n_categories, n_feature)
        self.pi = (1.0 / self.categories) * np.ones((self.categories,))  # (categories,) 每个高斯模型的权重

    def calc_mu_var(self):
        y_squeeze = np.squeeze(y_train)
        mu = []  # (n_categories, n_feature)
        var = []  # (n_categories, n_feature)
        for i in range(self.categories):
            X_i = X_train[y_squeeze == i, :]
            mu_i = np.mean(np.squeeze(X_i), axis=0)  # (n_feature, )
            var_i = np.var(np.squeeze(X_i), axis=0)  # (n_feature, )
            mu.append(mu_i)
            var.append(var_i)
        return np.array(mu), np.array(var)

    def update_w(self, X):
        n_samples = X.shape[0]
        pdfs = np.zeros((n_samples, self.categories))
        # print(self.mu[0])
        # print("=========================")
        # print(self.var[0])
        for i in range(self.categories):
            pdfs[:, i] = self.pi[i] * multivariate_normal.pdf(
                X, self.mu[i, :], np.diag(self.var[i]))  # （samples, 1）
            # print(pdfs[:, i])
        W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        return W

    def update_pi(self):
        pi = self.w.sum(axis=0) / self.w.sum()
        return pi

    def update_mu(self, X):
        mu = np.zeros((self.categories, self.n_features))
        for i in range(self.categories):
            mu[i] = np.average(X, axis=0, weights=self.w[:, i])  # w: (n_train_samples, categories)
        return mu

    def update_var(self, X):
        var = np.zeros((self.categories, self.n_features))
        for i in range(self.categories):
            var[i] = np.average((X - self.mu[i]) ** 2, axis=0, weights=self.w[:, i])
        return var

    def train(self, epochs, batch_size=10000):
        print("Start training...")
        for i in range(epochs):
            print(f"Epoch {i}... ")
            batch_num = int(self.n_train_samples / batch_size)
            for j in range(batch_num):
                right_bound = (i + 1) * batch_size
                if j == batch_num - 1:
                    right_bound = max(right_bound, self.n_train_samples)
                X = self.X_train[i * batch_size:right_bound, :]
                # E step
                self.w = self.update_w(X)
                self.pi = self.update_pi()
                # M step
                self.mu = self.update_mu(X)
                self.var = self.update_var(X)

            # loss = self.calc_loss()
            # if i % (epochs/100) == 0:
            #     print(f"Loss: {loss} at epoch {i}")
            # self.loss_list.append(loss)
        return

    def predict(self):
        print("Start predicting...")
        n_samples = self.X_test.shape[0]
        pdfs = np.zeros((n_samples, self.categories))
        for i in range(self.categories):
            pdfs[:, i] = self.pi[i] * multivariate_normal.pdf(
                self.X_test, self.mu[i, :], np.diag(self.var[i]))  # （samples, 1）
        labels = np.argmax(pdfs, axis=1)
        return labels

    def save_loss(self):
        dirname = "./log"
        filename = f"{self.model_type}_" + time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(dirname, filename)
        f = open(path, 'w')
        f.write("epoch,loss\n")
        for i, loss in enumerate(self.loss_list):
            # if i % 100 == 0:
            #     print(f"Epoch {i}, loss: {loss}")
            f.write(f"{i},{loss}\n")
        f.close()

    def calc_loss(self):
        pdfs = np.zeros((self.n_train_samples, self.categories))
        for i in range(self.categories):
            pdfs[:, i] = self.pi[i] * multivariate_normal.pdf(
                self.X_train, self.mu[i], np.diag(self.var[i]))
        return np.mean(np.log(pdfs.sum(axis=1)))

    def save_models(self):
        data = []
        path = f"./checkpoint/{self.model_type}.pth"
        for i in range(self.categories):
            data.append({
                "w": self.w,
                "pi": self.pi,
                "mu": self.mu,
                "var": self.var
            })
        f = open(path, 'wb')
        pickle.dump(data, f)
        f.close()

    def load_models(self):
        path = f"./checkpoint/{self.model_type}.pth"
        f = open(path, 'rb')
        data = pickle.load(f)
        self.w = data['w']
        self.pi = data['pi']
        self.mu = data['mu']
        self.var = data['var']
        f.close()


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(path="./data/preprocessed_data_1764.pkl", reload=True)
    # TODO: 把标准差var改成协方差矩阵
    packed_model = EM(X_train, y_train, X_test, categories=10)
    # packed_model.load_models()
    packed_model.train(epochs=200)
    predict_label = packed_model.predict()
    packed_model.save_models()
    # packed_model.save_loss()

    acc = np.mean(np.equal(predict_label, y_test))
    print(f"test accuracy: {acc}")
