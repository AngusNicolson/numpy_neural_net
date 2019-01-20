# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 19:28:30 2019

@author: angus
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
sns.set_style("whitegrid")


def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1, cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    if(file_name):
        plt.savefig(file_name)
        plt.close()

N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.1

X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)


make_plot(X, y, 'Data')


model = Sequential()

model.add(Dense(units=4, activation='relu', input_dim=2))
model.add(Dense(units=6, activation='relu'))
#model.add(Dense(units=6, activation='relu'))
#model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x=X_train, y=y_train, epochs=200)

model.evaluate(x=X_test, y=y_test)

plt.figure()
plt.plot(model.history.history['acc'])
plt.title('accuracy')
plt.show()

plt.figure()
plt.plot(model.history.history['loss'])
plt.title('loss')
plt.yscale('log')
plt.show()

