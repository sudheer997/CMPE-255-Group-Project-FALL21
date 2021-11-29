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
import time

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
        # measure data pre processing time
        start = time.time()
        text_vectorizer = vectorizer(vector_length=vector_length)
        X, y = text_vectorizer.fit_transform(data=formatted_data)
        print("************* Preprocessing the data Ended *****************")
        print("Pre processing Complete Sec time :", time.time() - start)
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
    # measure running time
    start = time.time()
    model_lr = LogisticRegressionModel.load_data("../data/News_Category_Dataset_v2.json",
                                                 vectorizer=Vectorizers.TFIDF_vectorizer.TFIDVectorizer,
                                                 vector_length=160000, save_model=True,
                                                 model_path_location="../models",
                                                 model_name="logistic_regression_tfidf.pkl")
    model_lr.get_model_performance()
    print("All Complete Sec time :", time.time() - start)

# Pre processing Complete Sec time : 740.432126045227
# All Complete Sec time : 1021.5989029407501
#               precision    recall  f1-score   support
#
#            0       0.66      0.34      0.45      1477
#            1       0.65      0.49      0.56      1872
#            2       0.70      0.39      0.50      1603
#            3       0.69      0.55      0.61      1107
#            4       0.86      0.68      0.76      1130
#            5       0.65      0.81      0.72      5162
#            6       0.76      0.77      0.77      2050
#            7       0.57      0.15      0.24      2000
#            8       0.84      0.71      0.77      1381
#            9       0.63      0.26      0.37      1115
#           10       0.55      0.68      0.61      2863
#           11       0.62      0.17      0.27      1272
#           12       0.75      0.91      0.82     10668
#           13       0.85      0.64      0.73      2049
#           14       0.78      0.69      0.73      1569
#           15       0.80      0.82      0.81      3182
#           16       0.76      0.81      0.78      3200
#           17       0.83      0.74      0.79      1205
#           18       0.58      0.85      0.69      5882
#           19       0.49      0.30      0.37      1099
#           20       0.81      0.52      0.64      1209
#
#     accuracy                           0.70     53095
#    macro avg       0.71      0.59      0.62     53095
# weighted avg       0.70      0.70      0.68     53095
#
