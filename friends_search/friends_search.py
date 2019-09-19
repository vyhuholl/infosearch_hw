import re
import os
import sys
import pickle
from math import log
from pymorphy2 import MorphAnalyzer
from collections import Counter
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


pattern = r"[а-яА-ЯёЁ]+\-?[а-яА-ЯёЁ]*"

m = MorphAnalyzer()
tokenizer = RegexpTokenizer(pattern)


def lemmatize(text):
    return ([m.parse(word)[0].normal_form
            for word in tokenizer.tokenize(text)])


class SearchEngine():
    def __init__(self, filepath):
        self.texts = dict()
        self.inv_index = defaultdict(list)
        self.preprocess(filepath)
        self.make_index()

    def preprocess(self, filepath):
        for root, dirs, files in os.walk(filepath):
            for name in files:
                with open(os.path.join(root, name), "r") as f:
                    self.texts[name[:-4]] = lemmatize(f.read())

    def make_index(self):
        for name in self.texts:
            for term in self.texts[name]:
                if name not in self.inv_index[term]:
                    self.inv_index[term].append(name)

    def tf_idf(self, term, document):
        tf = self.texts[document].count(term)
        idf = log(len(self.texts) / len(self.inv_index[term]))
        return tf * idf

    def search(self, query):
        result = defaultdict(float)
        for term in lemmatize(query):
            for document in self.inv_index[term]:
                result[document] += self.tf_idf(term, document)
        return [item[0] for item in
                sorted(result.items(), key=lambda kv: -kv[1])]


with open("FriendsSearch.pkl", "rb") as file:
    FriendsSearchEngine = pickle.load(file)

if __name__ == "__main__":
    for res in FriendsSearchEngine.search(sys.argv[1]):
        print(res)
