import numpy as np
from .metrics import r2_score
class LinearRegression():
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self.__thet = None       #__thet:Private variables

    def fit(self,X,y):
        assert len(X) == len(y), \
            "the length of x must be equal to the length of y"
        Xb = np.hstack((np.ones((X.shape[0], 1)), X))
        self.__thet = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(y)
        self.interception_ = self.__thet[0]
        self.coef_ = self.__thet[1:]
        return self

    def predict(self,X):
        assert self.coef_ is not None and self.interception_ is not None, \
            "must fit before predict"
        assert X.shape[1] == len(self.coef_), \
            "the mumber of fetures must be equal to the len of coef_"

        return np.hstack((np.ones((X.shape[0], 1)), X)).dot(self.__thet)


    def score(self,X,y):
        y_predict = self.predict(X)
        return r2_score(y,y_predict)

    def __repr__(self):
        return "LinearRegression()"