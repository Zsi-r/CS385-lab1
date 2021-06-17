# import numpy as np
# import pickle
#
# from utils import *
# from models.logistic_regression import LogisticRegression
# from models.logistic_regression_lasso import LogisticRegressionLasso
# from models.logistic_regression_ridge import LogisticRegressionRidge
#
#
# class PackedModel:
#
#     def __init__(self, X_train, y_train, X_test, categories, model_type):
#         self.n_train_samples, self.n_features = X_train.shape
#         self.categories = categories
#         self.y_in_each_model = self.init_y(y_train)
#         self.model_type = model_type
#
#         if model_type == LOGISTIC:
#             self.models = [LogisticRegression(X_train, self.y_in_each_model[i], X_test, i) for i in
#                            range(self.categories)]
#         elif model_type == RIDGE:
#             self.models = [LogisticRegressionRidge(X_train, self.y_in_each_model[i], X_test, i) for i in
#                            range(self.categories)]
#         elif model_type == LASSO:
#             self.models = [LogisticRegressionLasso(X_train, self.y_in_each_model[i], X_test, i) for i in
#                            range(self.categories)]
#
#     def init_y(self, y):
#         y_in_each_model = [np.zeros((self.n_train_samples, 1)) for i in range(self.categories)]
#         for i in range(self.n_train_samples):
#             cat = int(y[i])
#             # print(f"category: {cat}")
#             y_in_each_model[cat][i] = 1
#         return y_in_each_model
#
#     def save_models(self):
#         data = []
#         path = f"./checkpoint/{self.model_type}.pth"
#         for i in range(self.categories):
#             data.append({
#                 "w": self.models[i].w,
#                 "b": self.models[i].b
#             })
#         pickle.dump(data, open(path, 'wb'))
#
#     def load_models(self):
#         path = f"./checkpoint/{self.model_type}.pth"
#         data = pickle.load(open(path, 'rb'))
#         for i in range(self.categories):
#             self.models[i].w = data[i]['w']
#             self.models[i].b = data[i]['b']
#
#     def train(self, epochs, lr=0.1):
#         print("Start training...")
#         for i in range(self.categories):
#             self.models[i].train(epochs, lr)
#         return
#
#     def predict(self):
#         print("Start predicting...")
#         labels = []
#         for i in range(self.categories):
#             labels.append(self.models[i].predict())
#         labels = np.array(labels).T
#         labels = np.squeeze(labels)
#         print(labels.shape)
#         labels = np.argmax(labels, axis=1).reshape(-1, 1)
#         print(labels.shape)
#         return labels
#
#     def print_loss(self):
#         for i in range(self.categories):
#             print(f"Model {i}, loss_list: {self.models[i].loss_list}")
