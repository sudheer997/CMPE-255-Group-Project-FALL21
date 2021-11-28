# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:03:31 2021

@author: Checkout
"""
import re
import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn import svm
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC


class TFIDVectorizer:
    def __init__(self, vector_length=None):
        self.porter_stemmer = PorterStemmer()
        self.stop_words = stopwords.words("english")
        self.vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, ngram_range=(1, 2),
                                          max_features=vector_length, sublinear_tf=True)

    def preprocess_text(self, text):
        # Change all characters in the text to lower case
        text = text.lower()

        # Remove trailing spaces
        text = text.strip()

        # Remove special characters and numbers in the string
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(" \d+", " ", text)

        # Remove stop words and word has length less than 3
        tokenized_text = word_tokenize(text)
        processed_review = [token for token in tokenized_text if
                            token not in self.stop_words and len(token) >= 3 and not token.isdigit()]
        # Get the stem the word
        processed_review = " ".join(list(map(lambda x: self.porter_stemmer.stem(x), processed_review)))

        return processed_review

    def fit_transform(self, data=None):
        # Pre process the Head lines

        data['news'] = data[['headline', 'short_description']].agg(' '.join, axis=1)

        tqdm.pandas(desc="pre processing news articles")
        data["news"] = data["news"].progress_apply(lambda x: self.preprocess_text(x))

        # # pre process the short description
        # tqdm.pandas(desc="pre processing short description")
        # data["short_description"] = data["short_description"].progress_apply(lambda x: self.preprocess_text(x))
        #
        # # Combine the headline and short description
        # data["news"] = data["headline"] + data["short_description"]
        # Filter articles whose have more than 5 words per article
        data['words_length'] = data.news.apply(lambda i: len(i.split(" ")))
        data = data[data.words_length >= 5]
        data = data[data['category'].map(data['category'].value_counts()) > 3000]

        data.category = data.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)

        # Get vector representation of news articles using TfidfVectorizer
        # Chosen vector length on basis of Term frequency analysis
        # and we consider uni-grams and bi-grams for the vector-representation.
        X = self.vectorizer.fit_transform(data.news)


        y = data.category.values
        return X, y

    def transform(self, sentence):
        sentence = self.preprocess_text(sentence)
        return self.vectorizer.transform([sentence])




if __name__=="__main__":
        data_file= "./data/News_Category_Dataset_v2.json"
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
        
        vector_length=150000
        vectorizer=TFIDVectorizer

        print("************* Preprocessing the data started *****************")
        text_vectorizer = vectorizer(vector_length=vector_length)
        X, y = text_vectorizer.fit_transform(data=formatted_data)

        print("************* Preprocessing the data Ended *****************")
        X_train, X_test, y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.33,random_state=45,shuffle=True)

        print("************* model training started *****************")
        model = LinearSVC(random_state=0, tol=1e-5)
        model.fit(X_train, y_train.ravel())

        print("************* model training stopped *****************")

        # make predictions on test set
        y_pred = model.predict(X_test)

        # compute and print accuracy score
        print('Accuracy of LinearSVC: {0:0.2}'.format(accuracy_score(y_test, y_pred)))

        # Bagging Classifier
        from sklearn.ensemble import BaggingClassifier

        print("************* model training started *****************")

        model = BaggingClassifier(random_state=0, n_estimators=10)
        model.fit(X_train, y_train)

        print("************* model training stopped *****************")
        y_pred = model.predict(X_test)
        print('Accuracy of bagged KNN is : {0:0.2}'.format(accuracy_score(y_test, y_pred)))

        # Decision Tree Classifier
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()

        print("************* model training started *****************")
        model.fit(X_train, y_train)
        print("************* model training stopped *****************")

        y_pred = model.predict(X_test)
        print('The accuracy of Decision Tree is : {0:0.2}'.format(accuracy_score(y_test, y_pred)))

        # K Neighbors Classifier
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        print("************* model training started *****************")
        model.fit(X_train, y_train)
        print("************* model training stopped *****************")
        y_pred = model.predict(X_test)

        print('The accuracy of the KNN is : {0:0.2}'.format(accuracy_score(y_test, y_pred)))

        # Naive Bayesian
        from sklearn.naive_bayes import MultinomialNB

        model = MultinomialNB(alpha=0.1)
        print("************* model training started *****************")
        model.fit(X_train, y_train)
        print("************* model training stopped *****************")
        y_pred = model.predict(X_test)
        print('The accuracy of the naive baysian is : {0:0.2}'.format(accuracy_score(y_test, y_pred)))
