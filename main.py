import numpy as np

from utils import *
from preprocess import load_data
from models.logistic_regression import PackedLogisticRegression
from models.logistic_regression_ridge import PackedLogisticRegressionRidge
from models.logistic_regression_lasso import PackedLogisticRegressionLasso


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(path="./data/preprocessed_data.pkl", reload=True)

    # print(X_train.shape)  # (73257, 324)
    # print(y_train.shape)  # (73257, 1)
    # print(X_test.shape)  # (26032, 324)
    # print(y_test.shape)  # (26032, 1)
    # model_type = LOGISTIC
    # model_type = RIDGE
    # model_type = LASSO

    # packed_model = PackedLogisticRegression(X_train, y_train, X_test, categories=10)
    # packed_model = PackedLogisticRegressionRidge(X_train, y_train, X_test, categories=10)
    packed_model = PackedLogisticRegressionLasso(X_train, y_train, X_test, categories=10)

    # packed_model.load_models()
    packed_model.train(epochs=4, lr=0.1)
    packed_model.save_models()
    predict_label = packed_model.predict()
    packed_model.print_loss()

    acc = np.mean(np.equal(predict_label, y_test))
    print(f"test accuracy: {acc}")  # acc=0.72 when epochs=2000, lr=0.1


'''
    LogisticRegression epochs=2000, lr=0.1, acc=0.7208819913952059, Cross_entropy_loss=[0.6931471805599466, 0.3677869123414069, 0.28987923659913467, 0.262285911210369, 0.24998035963817292, 0.2436225172242931, 0.2399543612021582, 0.23762798899671506, 0.23601800634909587, 0.2348098460491107, 0.23383569602229612, 0.2330020124095653, 0.23225487904997907, 0.23156240928729382, 0.23090536271894113, 0.2302719450639237, 0.22965483538475037, 0.22904944349852582, 0.22845286759272992, 0.22786326038545335]
    LogisticRegressionRidge epochs=2000, lr=0.1, acc=0.7208435771358328, loss=[0.25000003304396473, 0.05831012173372958, 0.056661532430833966, 0.05497407904066535, 0.05330646998487959, 0.0517024105827623, 0.050195490142215686, 0.04880664593801381, 0.047544851239068216, 0.04640980923909576]
    LogisticRegressionLasso epochs=2000, lr=0.1, acc=
'''