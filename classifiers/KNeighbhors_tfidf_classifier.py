import os
import pickle
from abc import ABC
from copy import deepcopy, copy

import Vectorizers

from classifiers.classifier import Classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


class KNeighbhorsModel(Classifier, ABC):
    def __init__(self, X, y, text_vectorizer,
                 save_model=True,
                 model_path_location="../models",
                 model_name="Kneighbors_tfidf.pkl"):
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
        model = KNeighborsClassifier()
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
            return KNeighbhorsModel.load_data("../data/News_Category_Dataset_v2.json",
                                                     vectorizer=Vectorizers.TFIDF_vectorizer.TFIDVectorizer,
                                                     vector_length=160000, save_model=True,
                                                     model_path_location="../models",
                                                     model_name="Kneighbors_tfidf.pkl")

    def get_model_performance(self):
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))


if __name__ == "__main__":
    model_lr = KNeighbhorsModel.load_data("../data/News_Category_Dataset_v2.json",
                                                 vectorizer=Vectorizers.TFIDF_vectorizer.TFIDVectorizer,
                                                 vector_length=160000, save_model=True,
                                                 model_path_location="../models",
                                                 model_name="Kneighbors_tfidf.pkl")
    model_lr.get_model_performance()

#               precision    recall  f1-score   support
#
#            0       0.31      0.45      0.37      1477
#            1       0.35      0.46      0.40      1872
#            2       0.30      0.41      0.34      1603
#            3       0.43      0.54      0.48      1107
#            4       0.53      0.61      0.57      1130
#            5       0.52      0.73      0.61      5162
#            6       0.57      0.74      0.64      2050
#            7       0.25      0.28      0.26      2000
#            8       0.69      0.60      0.64      1381
#            9       0.42      0.22      0.29      1115
#           10       0.53      0.30      0.38      2863
#           11       0.31      0.20      0.24      1272
#           12       0.71      0.82      0.76     10668
#           13       0.70      0.48      0.57      2049
#           14       0.68      0.68      0.68      1569
#           15       0.78      0.64      0.70      3182
#           16       0.71      0.59      0.64      3200
#           17       0.79      0.56      0.66      1205
#           18       0.64      0.49      0.55      5882
#           19       0.43      0.18      0.26      1099
#           20       0.70      0.49      0.58      1209
#
#     accuracy                           0.58     53095
#    macro avg       0.54      0.50      0.51     53095
# weighted avg       0.59      0.58      0.57     53095