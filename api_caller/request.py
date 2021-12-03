# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 11:56:13 2021

@author: Checkout
"""

import requests
import json
import pandas as pd
if __name__=="__main__":
    url = 'http://localhost:5000/testApi'
    r = requests.post(url,json={"side": 8,
                                "bomb_count": 6,
                                "start": [2,2],
                                "goal":[7,7]})
    print(r.content)
    data = json.loads(r.text)
    pd.DataFrame.from_dict(eval(data))