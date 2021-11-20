import pandas as pd
import Vectorizers
from abc import ABCMeta, abstractmethod
import Vectorizers.TFIDF_vectorizer


class Classifier(object,  metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def load_data(cls, file_name, vectorizer=Vectorizers.TFIDF_vectorizer.TFIDVectorizer):
        pass

    @staticmethod
    def read_file(data_file):
        category = []
        headline = []
        short_description = []
        with open(data_file, "r+") as f:
            raw_data = f.readlines()

        for item in raw_data:
            item = item.replace("\n", "")
            item = eval(item)
            category.append(item.get("category", ""))
            headline.append(item.get("headline", ""))
            short_description.append(item.get("short_description", ""))
        formatted_data = pd.DataFrame({"headline": headline,
                                       "short_description": short_description,
                                       "category": category})
        return formatted_data

    @abstractmethod
    def train(self, save_model):
        pass

    @abstractmethod
    def predict(self, X):
        pass
