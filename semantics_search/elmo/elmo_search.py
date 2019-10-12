import sys
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
from pymorphy2 import MorphAnalyzer
from pymorphy2.tokenizers import simple_word_tokenize
from elmo_helpers import load_elmo_embeddings, get_elmo_vectors
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

tf.disable_v2_behavior()
tf.reset_default_graph()

with open("docs.pkl", "rb") as file:
    docs = pickle.load(file)

with open("vec.pkl", "rb") as file:
    vec = pickle.load(file)

m = MorphAnalyzer()


def lemmatize(text):
    return [m.parse(word)[0].normal_form
            for word in simple_word_tokenize(text)]


batcher, ids, elmo_input = load_elmo_embeddings(".")


class SearchELMO():
    def __init__(self, docs, vec):
        self.texts = np.array(docs)
        self.vec = vec

    def search(self, query, n=5):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            query_vec = np.transpose(np.mean(get_elmo_vectors(
                sess, [lemmatize(query)], batcher, ids, elmo_input),
                axis=1)).flatten()
            result = np.matmul(self.vec, query_vec)
            indices = np.argsort(result)[::-1].tolist()[:n]
        return list(zip(self.texts[indices], result[indices]))


ELMOSearchEngine = SearchELMO(docs, vec)


if __name__ == "__main__":
    for result in ELMOSearchEngine.search(sys.argv[1]):
        print(result)
