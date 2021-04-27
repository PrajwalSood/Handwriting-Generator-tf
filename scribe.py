# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 23:01:13 2021

@author: prajw
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data import DataSynthesis
from models import HandWritingPrediction, HandWritingSynthesis
from utils import plot_stroke


d = DataSynthesis()
generator = d.batch_generator(shuffle=False)
_, sentence, target = next(generator)

hws = HandWritingSynthesis()
hws.make_model(load_weights='models/trained/model_synthesis_overfit.h5')

sentence = tf.dtypes.cast(d.prepare_text('independent expert body'), float)
strokes, windows, phis, kappas = hws.infer(sentence, seed=18)

plot_stroke(strokes)
