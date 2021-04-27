import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        """
        X is a list of list of single characters

        X = [
            ["e", "l", "i", "a", "s"],
        ]
        """
        vocab = set()
        for x in X:
            vocab.update(set(x))
        self.vocab = dict(zip(vocab, range(1, len(vocab)+1)))
        self.inv_vocab = dict(zip(range(1, len(vocab)+1), vocab))
        return self

    def transform(self, X):
        """
        X is a list of strings
        """
        embeded = []
        for text in X:
            encoded = np.zeros((len(text), len(self.vocab)))
            for i, letter in enumerate(text):
                encoded[i][self.vocab[letter]-1] = 1
            embeded.append(encoded)

        return embeded

    def inverse_transform(self, s):
        """
        It does not really computes the full inverse transform
        It makes it easier to debug
        """
        s = np.array(s)
        sentences = []
        for encoded in s:
            inv = []
            for letter in encoded:
                if (letter == 0).all():
                    break
                inv.append(self.inv_vocab[letter.argmax()+1])
            sentences.append(inv)
        return sentences


if __name__ == '__main__':
    texts = [
        'welcome',
        'elwcome',
        'elias',
        'test',
    ]

    enc = OneHotEncoder()
    out = enc.fit_transform(texts)
    print(out)
