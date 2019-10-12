import sys
import pickle
import numpy as np
from pymorphy2 import MorphAnalyzer
from pymorphy2.tokenizers import simple_word_tokenize
from keras.models import Model
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer, load_vocabulary, get_checkpoint_paths

with open("docs.pkl", "rb") as file:
    docs = pickle.load(file)

with open("vec.pkl", "rb") as file:
    vec = pickle.load(file)

m = MorphAnalyzer()


def lemmatize(text):
    return [m.parse(word)[0].normal_form
            for word in simple_word_tokenize(text)]


class SearchBERT():
    def __init__(self, docs, vec):
        self.texts = np.array(docs)
        self.vec = vec
        paths = get_checkpoint_paths(".")
        inputs = load_trained_model_from_checkpoint(
            config_file=paths.config,
            checkpoint_file=paths.checkpoint, seq_len=50)
        outputs = MaskedGlobalMaxPool1D(name='Pooling')(inputs.output)
        self.model = Model(inputs=inputs.inputs, outputs=outputs)
        self.vocab = load_vocabulary(paths.vocab)
        self.tokenizer = Tokenizer(self.vocab)

    def search(self, query, n=5):
        tokens = self.tokenizer.tokenize(" ".join(lemmatize(query)))[:50]
        indices = [self.vocab[token] for token in tokens] + \
            [0 for i in range(50 - len(tokens))]
        segments = [0 for i in range(50)]
        query_vec = self.model.predict([np.array([indices]),
                                        np.array([segments])])[0]
        result = np.matmul(self.vec, query_vec)
        idxs = np.argsort(result)[::-1].tolist()[:n]
        return list(zip(self.texts[idxs], result[idxs]))


BERTSearchEngine = SearchBERT(docs, vec)


if __name__ == "__main__":
    for result in BERTSearchEngine.search(sys.argv[1]):
        print(result)
