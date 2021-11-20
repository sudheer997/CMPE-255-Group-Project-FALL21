import os
import pickle
from abc import ABC
from copy import deepcopy, copy

import Vectorizers

from classifiers.classifier import Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class LogisticRegressionModel(Classifier, ABC):
    def __init__(self, X, y, text_vectorizer,
                 save_model=True,
                 model_path_location="../models",
                 model_name="logistic_regression_tfidf.pkl"):
        super().__init__()
        self.X = X
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(y)
        self.vectorizer = text_vectorizer
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, self.y,
                                                                                stratify=y,
                                                                                test_size=0.33,
                                                                                random_state=45,
                                                                                shuffle=True)
        self.model = self.train(save_model)
        if save_model:
            self.persist(model_path_location, model_name)

    @classmethod
    def load_data(cls, data_file,
                  vectorizer=Vectorizers.TFIDF_vectorizer.TFIDVectorizer,
                  vector_length=150000,
                  **kwargs):

        formatted_data = cls.read_file(data_file)
        print("************* Preprocessing the data started *****************")
        text_vectorizer = vectorizer(vector_length=vector_length)
        X, y = text_vectorizer.fit_transform(data=formatted_data)
        print("************* Preprocessing the data Ended *****************")
        return cls(X, y, text_vectorizer, **kwargs)

    def train(self, save_model):
        print("************* model training started *****************")
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        print("************* model training stopped *****************")
        return model

    def persist(self, model_path_location, model_name):
        path = os.path.join(model_path_location, model_name)
        dummy_obj = copy(self)
        dummy_obj.X = None
        dummy_obj.y = None
        dummy_obj.X_train, dummy_obj.X_test, dummy_obj.y_train, dummy_obj.y_test = None, None, None, None
        with open(path, 'wb') as f:
            pickle.dump(dummy_obj, f)

    def predict(self, sentence):
        vectorized_sentence = self.vectorizer.transform(sentence)
        y_pred = self.model.predict(vectorized_sentence)
        return self.label_encoder.inverse_transform(y_pred)

    @classmethod
    def load_classifier(cls, train=False, model_path_location=None, model_name=None):
        path = os.path.join(model_path_location, model_name)
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if os.path.exists(path) and not train:
            with open(path, 'rb') as f:
                text_classifier = pickle.load(f)
            return text_classifier
        else:
            return LogisticRegressionModel.load_data("../data/News_Category_Dataset_v2.json",
                                                     vectorizer=Vectorizers.TFIDF_vectorizer.TFIDVectorizer,
                                                     vector_length=160000, save_model=True,
                                                     model_path_location="../models",
                                                     model_name="logistic_regression_tfidf.pkl")

    def get_model_performance(self):
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))


if __name__ == "__main__":
    model_lr = LogisticRegressionModel.load_data("../data/News_Category_Dataset_v2.json",
                                                 vectorizer=Vectorizers.TFIDF_vectorizer.TFIDVectorizer,
                                                 vector_length=160000, save_model=True,
                                                 model_path_location="../models",
                                                 model_name="logistic_regression_tfidf.pkl")
    model_lr.get_model_performance()


#                 precision    recall  f1-score   support
#
#           ARTS       0.38      0.15      0.21       498
# ARTS & CULTURE       0.42      0.10      0.16       442
#   BLACK VOICES       0.56      0.33      0.42      1494
#       BUSINESS       0.49      0.43      0.46      1959
#        COLLEGE       0.47      0.23      0.31       378
#         COMEDY       0.62      0.35      0.45      1708
#          CRIME       0.57      0.52      0.54      1124
# CULTURE & ARTS       0.64      0.13      0.21       340
#        DIVORCE       0.85      0.61      0.71      1131
#      EDUCATION       0.50      0.23      0.32       331
#  ENTERTAINMENT       0.53      0.80      0.64      5299
#    ENVIRONMENT       0.85      0.14      0.24       437
#          FIFTY       0.68      0.08      0.14       462
#   FOOD & DRINK       0.60      0.74      0.66      2055
#      GOOD NEWS       0.47      0.09      0.15       461
#          GREEN       0.46      0.30      0.36       865
# HEALTHY LIVING       0.48      0.16      0.24      2209
#  HOME & LIVING       0.72      0.66      0.69      1384
#         IMPACT       0.49      0.23      0.31      1141
#  LATINO VOICES       0.73      0.14      0.23       373
#          MEDIA       0.64      0.32      0.42       929
#          MONEY       0.54      0.25      0.34       563
#      PARENTING       0.48      0.66      0.55      2863
#        PARENTS       0.57      0.17      0.26      1305
#       POLITICS       0.64      0.88      0.74     10804
#   QUEER VOICES       0.77      0.62      0.69      2084
#       RELIGION       0.64      0.38      0.48       843
#        SCIENCE       0.68      0.37      0.48       719
#         SPORTS       0.71      0.65      0.68      1612
#          STYLE       0.68      0.06      0.10       744
# STYLE & BEAUTY       0.68      0.78      0.73      3184
#          TASTE       0.48      0.06      0.10       692
#           TECH       0.60      0.35      0.44       687
#  THE WORLDPOST       0.54      0.43      0.48      1209
#         TRAVEL       0.61      0.77      0.68      3263
#       WEDDINGS       0.81      0.70      0.75      1205
#     WEIRD NEWS       0.45      0.19      0.26       881
#       WELLNESS       0.48      0.85      0.62      5883
#          WOMEN       0.42      0.27      0.33      1152
#     WORLD NEWS       0.54      0.12      0.20       718
#      WORLDPOST       0.54      0.19      0.28       851
#
#       accuracy                           0.58     66282
#      macro avg       0.59      0.38      0.42     66282
#   weighted avg       0.59      0.58      0.55     66282
