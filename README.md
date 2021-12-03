# News Article Category Classification
## CMPE255-Group-Project-FALL21

Project by Group-5:
- Alavi A. Khan
- Jihyeok Choi
- Sirkara Mohana Sai Sachin Nekkanti
- Sudheer Tati

## Project Overview:
In these times there are a lot of sources that generate immense amounts of daily news on the Internet. In addition to that, the demand for information by people has been growing gradually, so it is very important that the news be classified so that users can access them quickly and effectively. We attempt to understand the current known categories can be used for personalizing news consumers based on past affinity towards certain news domains. In this project we have created an application that is capable of categorizing the news, given its headlines and short descriptions.

## Project Goal:
The goal is to build classifiers that take ‘news headlines’ and ‘short descriptions’ as inputs and provide the category most relevant to the news article. In this way, our classification model for news category classification could be used to identify the category of the current news. It is a multi-class classification problem.

## Data Overview:
The [dataset](https://www.kaggle.com/rmisra/news-category-dataset)  used for project contains around 200,000 news headlines from the year 2012 to 2018 obtained from HuffPost. The model trained on this dataset could be used to identify categories for untracked news articles used in different news articles.
It contains 42 categories of news articles such as Politics, Wellness , Entertainment , Travel and more than 30 relevant topics in recent times.

## Algorithms Considered/Used:
To classify news articles we have implemented following classification algorthims:
* LinearSVC
* Naive Bayesian
* Multi class logistic Regression
* Kth Nearest Neighbors
* Random Forest Classifier
* LSTM

To vectorization the data we have implemented following vectorization techniques:
* TF-IDF
* Word2Vec
* Tensorflow word embeddings


## Setting up project enviroment:
- Create new enviroment using annaconda or miniconda
```
conda create -n env_name pthon=3.6
```
- Install dependencies from requirements.txt
```
pip install -r requirements.txt
```
- After installating dependencies and all classifiers files are avaliable in [classifiers](https://github.com/sudheer997/CMPE-255-Group-Project-FALL21/tree/master/classifiers) directory and to run a train and run a model for example:

```
python classifiers/logistic_regression_tfidf_classifier.py
```



Classification of news articles based on headlines and short descriptions.

|Classification + tfidf|Accuracy Result|Running Time(Sec)|Classification + word2vec|Accuracy Result|Running Time(Sec)|
|---|---|---|---|---|---|
|1. Random Forest                    | 0.64|1562|6. logistic Regression           | 0.70|667|
|2. Naive Bayesian                   | 0.67|760|7. Linear SVC                     | 0.63|3844|
|3. logistic Regression              | 0.62|1021|8. K Neigbhors                    | 0.59|770|
|4. Linear SVC                      | 0.72|180|9. Random Forest                   |0.59|1562|
|5. K Neighbhors                    | 0.58|346||


## To host Flask Jinja web-app:
- Run flask_app.py from command line with current working directory as project root directory
```
python web_app/flask_app.py
```
- Hosted app will be on http://localhost:5000/main port.

