# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:22:47 2021

@author: Checkout
"""

from flask import Flask, request, jsonify



app = Flask(__name__)

@app.route('/testApi',methods=['POST'])

def run_sim():
    data= request.get_json(force=True)
    print(data.content)
    #print(df_pol)
    #print(df_pol.to_json())
    return jsonify(data.content)
    


if __name__ == "__main__" :
    app.run(port=5000,debug=True)
    