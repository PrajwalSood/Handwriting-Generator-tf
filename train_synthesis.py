import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data import DataSynthesis
from utils import plot_stroke
from models.handwriting_synthesis import HandWritingSynthesis

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''CONFIG'''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


MODEL_PATH = 'models/trained/model_synthesis_overfit.h5'
EPOCH_MODEL_PATH = 'models/trained/model_synthesis_overfit_{}.h5'
LOAD_PREVIOUS = None
DATA_PATH = 'data/strokes-py3.npy'

VERBOSE = False

model_kwargs = {
    'regularizer_type': 'l2',
    'reg_mean': 0.,
    'reg_std': 0.,
    'reg_l2': 0.,
    'lr': .0001,
    'rho': .95,
    'momentum': .9,
    'epsilon': .0001,
    'centered': True,
    'inf_type': 'max',
}

HIDDEN_DIM = 400
NUM_LAYERS = 3

data_kwargs = {
    'path_to_data': DATA_PATH,
}

train_generator_kwargs = {
    'batch_size': 1,
    'shuffle': False,
}

EPOCHS = 2000
STEPS_PER_EPOCH = 1
MODEL_CHECKPOINT = 2

# bias for writing ~~style~~
BIAS = None


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''TRAIN'''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

D = DataSynthesis(**data_kwargs)
WINDOW_SIZE = len(D.sentences[0][0])
CHAR_LENGTH = D.char_length

model_kwargs['vocab_size'] = WINDOW_SIZE
# minus 1 because we take one element out for input and target
model_kwargs['char_length'] = CHAR_LENGTH
hws = HandWritingSynthesis(**model_kwargs)
hws.make_model(load_weights=LOAD_PREVIOUS)

nan = False
generator = D.batch_generator(
    **train_generator_kwargs,
)

input_states = [
    # stateh1, statec1
    tf.zeros((1, HIDDEN_DIM), dtype=float), tf.zeros((1, HIDDEN_DIM), dtype=float),
    # window kappa
    tf.zeros((1, WINDOW_SIZE), dtype=float), tf.zeros((1, 10), dtype=float),
    # phi
    tf.zeros((1, CHAR_LENGTH + 1), dtype=float),
    # sentence
    None,
    # stateh2, statec2
    tf.zeros((1, HIDDEN_DIM), dtype=float), tf.zeros((1, HIDDEN_DIM), dtype=float),
    # stateh3, statec3
    tf.zeros((1, HIDDEN_DIM), dtype=float), tf.zeros((1, HIDDEN_DIM), dtype=float),
]
hws.model.load_weights(MODEL_PATH)
try:
    # Test for overfitting
    strokes, sentence, targets = next(generator)
    for e in range(1, EPOCHS + 1):
        train_loss = []
        for s in tqdm(range(1, STEPS_PER_EPOCH+1), desc="Epoch {}/{}".format(e, EPOCHS)):
            # strokes, sentence, targets = next(generator)
            input_states[5] = sentence
            loss = hws.train([strokes, input_states], targets)
            train_loss.append(loss)

            if loss is np.nan:
                nan = True
                print('exiting train @epoch : {}'.format(e))
                break

        mean_loss = np.mean(train_loss)
        print("Epoch {:03d}: Loss: {:.3f}".format(e, mean_loss))

        if e % 1 == 0:
            hws.model.save_weights(EPOCH_MODEL_PATH.format(e))

        if nan:
            break

except KeyboardInterrupt:
    pass

if not nan:
    hws.model.save_weights(MODEL_PATH)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''''''EVALUATE'''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

verbose_sentence = "".join(D.encoder.inverse_transform(sentence)[0])
strokes1, _, _, _ = hws.infer(sentence, inf_type='max', verbose=verbose_sentence)
plot_stroke(strokes1)
import ipdb; ipdb.set_trace()
