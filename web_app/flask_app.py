# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 15:02:16 2021

@author: Checkout
"""

from flask import Flask, redirect, url_for, render_template, request
import pickle
import Vectorizers.TFIDF_vectorizer as vectorizer_class
vector_length=150000
#vectorizer=vectorizer_class.TFIDVectorizer
#text_vectorizer = vectorizer(vector_length=vector_length)
path="./clf_model/SVC_tfidf.pkl"
#import pickle
with open(path, 'rb') as f:
    clf=pickle.load(f)
#    y_pred = clf.predict(X_test)
path="./clf_model/label_encoder.pkl"
#import pickle
with open(path, 'rb') as f:
    label_encoder=pickle.load(f)

path="./clf_model/text_vectorizer_tfidf.pkl"
#import pickle
with open(path, 'rb') as f:
    text_vectorizer=pickle.load(f)

        


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("base_extend2.html")

@app.route("/login", methods=["POST", "GET"])
def main():
    if request.method == "POST":
        nd = request.form["nm"]
        #print("wowo"+nd)
        sentence=nd
        vectorized_sentence = text_vectorizer.transform(sentence=sentence)
        y_pred = clf.predict(vectorized_sentence)
        label_encoder.inverse_transform(y_pred)[0]

        return redirect(url_for("user", usr=nd))
    else:
        return render_template("main.html")

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"

if __name__ == "__main__":
    app.run(debug=True)