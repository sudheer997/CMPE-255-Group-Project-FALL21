import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


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
        tqdm.pandas(desc="pre processing headline")
        data["headline"] = data["headline"].progress_apply(lambda x: self.preprocess_text(x))

        # pre process the short description
        tqdm.pandas(desc="pre processing short description")
        data["short_description"] = data["short_description"].progress_apply(lambda x: self.preprocess_text(x))

        # Combine the headline and short description
        data["news"] = data["headline"] + data["short_description"]

        # Get vector representation of news articles using TfidfVectorizer
        # Chosen vector length on basis of Term frequency analysis
        # and we consider uni-grams and bi-grams for the vector-representation.

        X = self.vectorizer.fit_transform(data.news)
        y = data.category.values
        return X, y

    def transform(self, sentence):
        sentence = self.preprocess_text(sentence)
        return self.vectorizer.transform([sentence])


