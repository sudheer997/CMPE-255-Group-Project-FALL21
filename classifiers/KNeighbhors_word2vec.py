import os
import pickle
from abc import ABC
from copy import copy

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import Vectorizers
from Vectorizers.word2vec_vectorizer import Word2Vec_vectorizer
from classifiers.classifier import Classifier
from sklearn.ensemble import RandomForestClassifier

class randomforestModel(Classifier, ABC):
    def __init__(self, X, y, text_vectorizer,
                 save_model=True,
                 model_path_location="../models",
                 model_name="randomforest_word2vec.pkl"):
        super().__init__()
        self.X = X
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(y)
        self.vectorizer = text_vectorizer
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                stratify=y,
                                                                                test_size=0.20,
                                                                                random_state=45,
                                                                                shuffle=True)
        self.model = self.train(save_model)
        if save_model:
            self.persist(model_path_location, model_name)

    @classmethod
    def load_data(cls, data_file,
                  vectorizer=None,
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
        model = RandomForestClassifier()
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
            return randomforestModel.load_data("../data/News_Category_Dataset_v2.json",
                                                     vectorizer=Vectorizers.TFIDF_vectorizer.TFIDVectorizer,
                                                     vector_length=160000, save_model=True,
                                                     model_path_location="../models",
                                                     model_name="randomforest_word2vec.pkl")

    def get_model_performance(self):
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))


if __name__ == "__main__":
    model_lr = randomforestModel.load_data("../data/News_Category_Dataset_v2.json",
                                                 vectorizer=Word2Vec_vectorizer,
                                                 vector_length=160000, save_model=True,
                                                 model_path_location="../models",
                                                 model_name="randomforest_word2vec.pkl")
    model_lr.get_model_performance()

#                 precision    recall  f1-score   support
#
#   BLACK VOICES       0.66      0.13      0.22       895
#       BUSINESS       0.53      0.30      0.38      1135
#         COMEDY       0.58      0.16      0.25       972
#          CRIME       0.56      0.53      0.55       670
#        DIVORCE       0.73      0.47      0.57       685
#  ENTERTAINMENT       0.49      0.75      0.59      3129
#   FOOD & DRINK       0.65      0.69      0.67      1242
# HEALTHY LIVING       0.53      0.06      0.10      1212
#  HOME & LIVING       0.72      0.44      0.54       837
#         IMPACT       0.46      0.08      0.14       676
#      PARENTING       0.44      0.59      0.50      1735
#        PARENTS       0.50      0.08      0.14       771
#       POLITICS       0.67      0.89      0.77      6466
#   QUEER VOICES       0.70      0.38      0.49      1242
#         SPORTS       0.69      0.33      0.44       951
# STYLE & BEAUTY       0.74      0.72      0.73      1928
#         TRAVEL       0.60      0.72      0.65      1939
#       WEDDINGS       0.82      0.55      0.66       730
#       WELLNESS       0.50      0.80      0.62      3565
#          WOMEN       0.49      0.16      0.24       666
#      WORLDPOST       0.72      0.34      0.46       733
#
#       accuracy                           0.59     32179
#      macro avg       0.61      0.44      0.46     32179
#   weighted avg       0.60      0.59      0.55     32179