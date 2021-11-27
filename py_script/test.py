# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:03:31 2021

@author: Checkout
"""
import pandas as  pd
import Vectorizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
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
        
        vectorizer=Vectorizers.TFIDF_vectorizer.TFIDVectorizer
        print("************* Preprocessing the data started *****************")
        text_vectorizer = vectorizer(vector_length=vector_length)
        X, y = text_vectorizer.fit_transform(data=formatted_data)
        print("************* Preprocessing the data Ended *****************")
        
        label_encoder = LabelEncoder()
        
        y = label_encoder.fit_transform(y)
        
        
        X_train, X_test, y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.33,random_state=45,shuffle=True)
        print("************* model training started *****************")
        #model = LogisticRegression()
        model = svm.SVC()
        model.fit(X_train, y_train,verbose=True)
        print("************* model training stopped *****************")
        
        
        
        
        
        
        