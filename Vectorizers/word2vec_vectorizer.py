from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import re
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from tqdm import tqdm


class Word2Vec_vectorizer:
    def __init__(self, vector_length=None):
        self.model = None
        self.porter_stemmer = PorterStemmer()
        self.stop_words = stopwords.words("english")

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
        return processed_review

    def word2_vec_vectorizer(self, sentence):
        if len(sentence) == 0:
            print(sentence)
        sentence_vector = self.model.wv[sentence].mean(axis=0)
        return sentence_vector

    def fit_transform(self, data=None):
        # combine headline and short description
        data['news'] = data[['headline', 'short_description']].agg(' '.join, axis=1)

        # Preprocess the news articles
        tqdm.pandas(desc="pre processing news article text")
        data["processed_text"] = data["news"].progress_apply(lambda x: self.preprocess_text(x))

        # Filter articles whose have more than 5 words per article
        data['words_length'] = data.processed_text.apply(lambda i: len(i))
        data = data[data.words_length >= 5]
        data.words_length.describe()

        # Build word2Vec model
        self.model = Word2Vec(data.processed_text, min_count=1, vector_size=300)

        tqdm.pandas(desc="Converting text to vector using Word2Vec")
        data["sentence_vector"] = data["processed_text"].progress_apply(lambda sentence: self.word2_vec_vectorizer(
            sentence))
        data = data[data['category'].map(data['category'].value_counts()) > 3000]

        X = np.vstack(data.sentence_vector.values)

        data.category = data.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
        y = data.category.values

        return X, y

    def transform(self, sentence):
        sentence = self.preprocess_text(sentence)
        return self.word2_vec_vectorizer(sentence)
