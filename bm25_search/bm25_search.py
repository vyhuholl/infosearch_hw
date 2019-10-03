import sys
import pickle
import numpy as np
from collections import Counter
from collections import defaultdict
from scipy.sparse import csr_matrix
from pymorphy2 import MorphAnalyzer
from pymorphy2.tokenizers import simple_word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

m = MorphAnalyzer()


def lemmatize(text):
    return [m.parse(word)[0].normal_form
            for word in simple_word_tokenize(text)]


class SearchEngineBM25():
    def __init__(self, data, b=0.75, k=2):
        self.texts = np.array(data)
        self.b = b
        self.k = k
        self.vectorizer = CountVectorizer(tokenizer=lemmatize)
        tf = self.vectorizer.fit_transform(self.texts)
        N = (tf.getnnz(axis=1)).sum()
        lens = np.array(tf.sum(axis=0))[0]
        self.avgdl = lens.mean()
        lens_rel = np.array([lens[x] / self.avgdl for x in tf.nonzero()[1]])
        idf = np.array([np.log((N - y + 0.5) / (y + 0.5))
                        for y in tf.nonzero()[0]])
        bm25 = idf * tf.data * (k + 1) / (tf.data + k * (1 - b + b * lens_rel))
        self.matrix = csr_matrix((bm25, tf.indices, tf.indptr), shape=tf.shape)
        self.matrix = self.matrix.transpose(copy=True)
        self.tf = []
        df = defaultdict(int)
        for text in self.texts:
            self.tf.append(Counter(lemmatize(text)))
            for word in set(lemmatize(text)):
                df[word] += 1
        self.idf = {key: np.log((N - val + 0.5) / (val + 0.5))
                    for key, val in df.items()}

    def search_naive(self, query, n=5):
        query = lemmatize(query)
        result = []
        for document in self.tf:
            score = 0
            for word in query:
                if word in document:
                    score += self.idf[word] * document[word] * (self.k + 1) / \
                             (document[word] + self.k * (1 - self.b + self.b *
                                                         len(document) /
                                                         self.avgdl))
            result.append(score)
        return sorted(list(zip(self.texts, result)),
                      key=lambda x: x[1], reverse=True)[:n]

    def search_matmul(self, query, n=5):
        query_vec = self.vectorizer.transform([query])
        result = np.array((query_vec * self.matrix).todense())[0]
        indices = np.argsort(result)[::-1].tolist()[:n]
        return list(zip(self.texts[indices], result[indices]))


with open("BMSearchEngine.pkl", "rb") as file:
    BMSearchEngine = pickle.load(file)

if __name__ == "__main__":
    for result in BMSearchEngine.search_matmul(sys.argv[1]):
        print(result)
