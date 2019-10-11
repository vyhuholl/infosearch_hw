import pickle
import numpy as np
from pymorphy2 import MorphAnalyzer
from pymorphy2.tokenizers import simple_word_tokenize
from gensim.models.keyedvectors import KeyedVectors

with open("docs.pkl", "rb") as file:
    docs = pickle.load(file)

m = MorphAnalyzer()


def lemmatize(text):
    return [m.parse(word)[0].normal_form
            for word in simple_word_tokenize(text)]


class SearchFasttext():
    def __init__(self, data, model_file):
        self.texts = np.array(data)
        self.model = KeyedVectors.load(model_file)
        self.vec = np.zeros((len(self.texts), self.model.vector_size))
        for i in range(len(self.texts)):
            lemmas = lemmatize(self.texts[i])
            lemmas_vectors = np.zeros((len(lemmas), self.model.vector_size))
            for idx, lemma in enumerate(lemmas):
                if lemma in self.model.vocab:
                    lemmas_vectors[idx] = self.model[lemma]
            if lemmas_vectors.shape[0] is not 0:
                self.vec[i] = np.mean(lemmas_vectors, axis=0)

    def search(self, query, n=5):
        lemmas = lemmatize(query)
        query_vec = np.zeros((self.model.vector_size, ))
        lemmas_vectors = np.zeros((len(lemmas), self.model.vector_size))
        for idx, lemma in enumerate(lemmas):
            if lemma in self.model.vocab:
                lemmas_vectors[idx] = self.model[lemma]
        if lemmas_vectors.shape[0] is not 0:
            query_vec = np.mean(lemmas_vectors, axis=0)
        query_vec = np.transpose(query_vec)
        result = np.matmul(self.vec, query_vec)
        indices = np.argsort(result)[::-1].tolist()[:n]
        return list(zip(self.texts[indices], result[indices]))


FasttextSearchEngine = SearchFasttext(docs, "model.model")


if __name__ == "__main__":
    for result in FasttextSearchEngine.search(sys.argv[1]):
        print(result)
