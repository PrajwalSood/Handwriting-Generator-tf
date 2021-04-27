import re
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from utils import OneHotEncoder


class Data(ABC):

    def __init__(
        self,
        path_to_data='data/strokes-py3.npy',
        path_to_sentences='data/sentences.txt',
        clean_text=True,
        allow_multiple=False,
    ):
        self.path_to_data = path_to_data
        self.path_to_sentences = path_to_sentences
        self.clean_text = clean_text
        self.encoder = OneHotEncoder()
        # padd with one, to have the pen lifted
        self.padding = [[1, 0, 0]]

    @property
    def strokes(self):
        if not hasattr(self, '_strokes'):
            strokes = np.load(self.path_to_data, allow_pickle=True)
            self._max_length = max(map(len, strokes))
            self._strokes = strokes
        return self._strokes.copy()

    @property
    def max_length(self):
        if not hasattr(self, '_max_length'):
            _ = self.strokes
        return self._max_length

    def prepare_text(self, text):
        _ = self.sentences
        text = re.sub('[^.,a-zA-Z!?\-\'" \n]', '#', text)
        text = text.split('\n')
        text = self.encoder.transform(text)[0]
        text = np.vstack((
            text,
            self.char_padding*(self.char_length-text.shape[0])
        ))
        return tf.dtypes.cast(text.reshape((1,) + text.shape), float)

    @property
    def sentences(self):
        if not hasattr(self, '_sentences'):
            with open(self.path_to_sentences) as f:
                texts = f.read()

            if self.clean_text:
                texts = re.sub('[^.,a-zA-Z!?\-\'" \n]', '#', texts)

            texts = texts.split('\n')
            self._sentences = self.encoder.fit_transform(texts)
            self._char_length = max(map(len, self._sentences))
            self.char_padding = [[0] * len(self._sentences[0][0])]
        return self._sentences.copy()

    @property
    def char_length(self):
        if not hasattr(self, '_char_length'):
            _ = self.sentences
        return self._char_length

    @abstractmethod
    def batch_generator(self, sequence_lenght, batch_size=10):
        raise NotImplementedError


class DataPrediction(Data):

    def __init__(self, path_to_data='data/strokes-py3.npy'):
        super(DataPrediction, self).__init__(path_to_data=path_to_data)

    def batch_generator(self, sequence_lenght, batch_size=10):
        # We want (x3, x1, x2) --> (x1, x2, x3)
        all_strokes = tf.gather(np.vstack(self.strokes), [1, 2, 0], axis=1)
        while True:
            batch_strokes = []
            batch_targets = []
            for _ in range(batch_size):
                strokes = tf.image.random_crop(all_strokes, (sequence_lenght+1, 3))
                # batch_strokes.append(tf.concat((tf.zeros((1, 3)), strokes[:-1, :]), axis=0))
                batch_strokes.append(strokes[:-1, :])
                batch_targets.append(strokes[1:, :])
            yield tf.stack(batch_strokes), tf.stack(batch_targets)


class DataSynthesis(Data):

    def __init__(
        self,
        path_to_sentences='data/sentences.txt',
        clean_text=True,
        path_to_data='data/strokes-py3.npy',
    ):
        super(DataSynthesis, self).__init__(
            path_to_data=path_to_data,
            path_to_sentences=path_to_sentences,
            clean_text=clean_text,
        )

    def batch_generator(self, batch_size=1, shuffle=True):
        all_strokes = self.strokes
        all_sentences = self.sentences
        idx = np.arange(0, len(all_sentences))
        while True:
            if shuffle:
                np.random.shuffle(idx)

            batch_strokes = []
            batch_sentences = []
            batch_targets = []
            for it in idx:
                strokes, sentences = all_strokes[it], all_sentences[it]
                sentences = np.vstack((
                    sentences,
                    self.char_padding*(self.char_length-sentences.shape[0])
                ))
                # We want (x3, x1, x2) --> (x1, x2, x3)
                strokes = tf.gather(strokes, [1, 2, 0], axis=1)
                batch_strokes.append(strokes[:-1, :])
                batch_sentences.append(sentences)
                batch_targets.append(strokes[1:, :])
                if len(batch_strokes) == batch_size:
                    yield (
                        tf.dtypes.cast(tf.stack(batch_strokes), dtype=float),
                        tf.dtypes.cast(tf.stack(batch_sentences), dtype=float),
                        tf.dtypes.cast(tf.stack(batch_targets), dtype=float),
                    )
                    batch_strokes = []
                    batch_sentences = []
                    batch_targets = []
