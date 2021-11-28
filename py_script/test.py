# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:03:31 2021

@author: Checkout
"""
import pandas as  pd
import Vectorizers.TFIDF_vectorizer as vecrorizer_class
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
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
        
        vectorizer=vecrorizer_class.TFIDVectorizer
        print("************* Preprocessing the data started *****************")
        text_vectorizer = vectorizer(vector_length=vector_length)
        X, y = text_vectorizer.fit_transform(data=formatted_data)
        print("************* Preprocessing the data Ended *****************")
        path="./models/text_vectorizer_tfidf.pkl"
        #import pickle
        with open(path, 'wb') as f:
            pickle.dump(text_vectorizer, f)
        
        label_encoder = LabelEncoder()
        path="./models/label_encoder.pkl"
        label_encoder.fit(y)
        label_encoder.classes_
        y = label_encoder.fit_transform(y)
        label_encoder.fit(y)
        
        with open(path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        
        X_train, X_test, y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.33,random_state=45,shuffle=True)
        print("************* model training started *****************")
        #model = LogisticRegression()
        
        
        
        #model = svm.SVC(kernel="linear",verbose=True)
        model=LinearSVC(random_state=0,tol=1e-5)
        model.fit(X_train, y_train)
        print("************* model training stopped *****************")
        path="./models/SVC_tfidf.pkl"
        #import pickle
        with open(path, 'wb') as f:
            pickle.dump(model, f)
            
        # make predictions on test set
        y_pred = model.predict(X_test)

        # compute and print accuracy score
        print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
        
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        
        path="./models/dt_tfidf.pkl"
        #import pickle
        with open(path, 'wb') as f:
            pickle.dump(clf, f)
        y_pred = clf.predict(X_test)

        # compute and print accuracy score
        print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
        
        ovo = OneVsOneClassifier(model)
        # fit model
        ovo.fit(X_train, y_train)
        # make predictions
        y_pred = ovo.predict(X_test)

        # compute and print accuracy score
        print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
        sentence="Record of Natural Disaster"
        vectorized_sentence = text_vectorizer.transform(sentence=sentence)
        y_pred = model.predict(vectorized_sentence)
        label_encoder.inverse_transform(y_pred)[0]
        LabelEncoder.classes__
        
        
        
        