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
from sklearn.tree import DecisionTreeClassifier

class decisiontreeModel(Classifier, ABC):
    def __init__(self, X, y, text_vectorizer,
                 save_model=True,
                 model_path_location="../models",
                 model_name="decision_tree_word2vec.pkl"):
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
        model = DecisionTreeClassifier()
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
            return decisiontreeModel.load_data("../data/News_Category_Dataset_v2.json",
                                                     vectorizer=Vectorizers.TFIDF_vectorizer.TFIDVectorizer,
                                                     vector_length=160000, save_model=True,
                                                     model_path_location="../models",
                                                     model_name="decision_tree_word2vec.pkl")

    def get_model_performance(self):
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))


if __name__ == "__main__":
    model_lr = decisiontreeModel.load_data("../data/News_Category_Dataset_v2.json",
                                                 vectorizer=Word2Vec_vectorizer,
                                                 vector_length=160000, save_model=True,
                                                 model_path_location="../models",
                                                 model_name="decision_tree_word2vec.pkl")
    model_lr.get_model_performance()

#                 precision    recall  f1-score   support
#
#   BLACK VOICES       0.14      0.14      0.14       895
#       BUSINESS       0.22      0.23      0.23      1135
#         COMEDY       0.17      0.17      0.17       972
#          CRIME       0.34      0.33      0.33       670
#        DIVORCE       0.31      0.32      0.31       685
#  ENTERTAINMENT       0.40      0.40      0.40      3129
#   FOOD & DRINK       0.45      0.46      0.46      1242
# HEALTHY LIVING       0.15      0.15      0.15      1212
#  HOME & LIVING       0.32      0.33      0.32       837
#         IMPACT       0.08      0.08      0.08       676
#      PARENTING       0.30      0.30      0.30      1735
#        PARENTS       0.14      0.14      0.14       771
#       POLITICS       0.67      0.66      0.67      6466
#   QUEER VOICES       0.27      0.27      0.27      1242
#         SPORTS       0.24      0.24      0.24       951
# STYLE & BEAUTY       0.54      0.54      0.54      1928
#         TRAVEL       0.44      0.46      0.45      1939
#       WEDDINGS       0.40      0.41      0.40       730
#       WELLNESS       0.46      0.45      0.45      3565
#          WOMEN       0.12      0.11      0.12       666
#      WORLDPOST       0.28      0.29      0.29       733
#
#       accuracy                           0.40     32179
#      macro avg       0.31      0.31      0.31     32179
#   weighted avg       0.40      0.40      0.40     32179
