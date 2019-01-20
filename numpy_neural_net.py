# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 19:34:06 2019

@author: angus
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
sns.set_style("whitegrid")

def relu(Z):
    return np.maximum(0, Z)

def relu_back(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[dZ <= 0] = 0
    return dZ

def sigmoid(Z):
    return 1/(1+np.exp(-Z))  

def sigmoid_back(dA, Z):
    sig = sigmoid(Z)
    dZ = dA * sig*(1-sig)
    return dZ
    
def tanh(Z):
    return np.tanh(Z)

def tanh_back(dA, Z):
    return dA * (1- tanh(Z)**2)

def leaky_relu(Z, a=0.01):
    if Z < 0:
        return a*Z
    else:
        return Z

def leaky_relu_back(dA, Z, a=0.01):
    dZ = np.array(dA, copy=True)
    dZ[dZ <= 0] = dA * a
    return dZ


def init_params(nn_architecture, seed=1964):
    np.random.seed(seed)
    param_values = []
    scale = 1.0
    
    for layer in nn_architecture:
        W = np.random.rand(layer['output_dim'], layer['input_dim']) * scale - (0.5 * scale)
        b = np.random.rand(layer['output_dim'], 1) * scale - (0.5 * scale)
            
        param_values.append({'W': W, 'b': b})
    
    return param_values

def single_layer_forward(A_prev, W_curr, b_curr, activation):
    
    if activation == 'relu':
        activation_func = relu
    elif activation == 'tanh':
        activation_func = tanh
    elif activation == 'leaky_relu':
        activation_func = leaky_relu
    elif activation == 'sigmoid':
        activation_func = sigmoid
    else:
        raise Exception('Activation function not supported')
    
    
    U_curr = np.dot(W_curr, A_prev)
    Z_curr = U_curr + b_curr
    A_curr = activation_func(Z_curr)
    
    return Z_curr, A_curr

def forward_pass(X, param_values, nn_architecture):
    memory = [{'A': X, 'Z':None}]
    #Problem in this loop I think. Maybe with the single_layer_forward function
    for i, layer in enumerate(nn_architecture):
        Z, A = single_layer_forward(memory[i]['A'], param_values[i]['W'], param_values[i]['b'], nn_architecture[i]['activation'])
        memory.append({'A': A, 'Z': Z})
        
        i+=1
    
    return memory

def binary_cross_entropy(y_hat, y):
    return -(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def binary_cross_entropy_back(Y_hat, Y):
    return -np.divide(Y,Y_hat) + np.divide(1-Y, 1-Y_hat)

def get_accuracy(y_hat, y):
    y_hat = np.around(y_hat)
    
    return (y_hat==y).mean()

def single_layer_back(A_prev, Z_curr, W_curr, b_curr, dA_curr, activation):
    
    if activation == 'relu':
        back_activation_func = relu_back
    elif activation == 'tanh':
        back_activation_func = tanh_back
    elif activation == 'leaky_relu':
        back_activation_func = leaky_relu_back
    elif activation == 'sigmoid':
        back_activation_func = sigmoid_back
    else:
        raise Exception('Activation function not supported')
    
    m = len(dA_curr)
    dZ_curr = back_activation_func(dA_curr, Z_curr)
    db_curr = (1/m)*np.sum(dZ_curr, axis=1, keepdims=True)
    dA_prev = np.dot(W_curr.T, dZ_curr)
    dW_curr = (1/m)*np.dot(dZ_curr, A_prev.T)
    
    return dA_prev, db_curr, dW_curr

def back_pass(Y, memory, param_values, nn_architecture):
    grad_values = [dict() for x in range(len(nn_architecture))]
    Y_hat = memory[-1]['A']
    Y = Y.reshape(Y_hat.shape)
    
    dA_prev = binary_cross_entropy_back(Y_hat, Y)
    
    for i, layer in enumerate(nn_architecture):
        dA_curr = dA_prev
        layer_i = - i - 1
        dA_prev, db_curr, dW_curr = single_layer_back(memory[layer_i - 1]['A'],
                                                      memory[layer_i]['Z'],
                                                      param_values[layer_i]['W'],
                                                      param_values[layer_i]['b'],
                                                      dA_curr,
                                                      nn_architecture[layer_i]['activation'])
        
        grad_values[layer_i].update({'db': db_curr, 'dW': dW_curr})
    
    return grad_values

def update_params(grad_values, param_values, nn_architecture, learning_rate):
    #print(param_values[0]['b'][0])
    for i, layer in enumerate(nn_architecture):
        param_values[i] = {'W': param_values[i]['W'] - grad_values[i]['dW'] * learning_rate,
                                'b': param_values[i]['b'] - grad_values[i]['db'] * learning_rate}
    #print(param_values[0]['b'][0])
    #print()
    return param_values

def train(X, Y, nn_architecture, epochs, learning_rate, seed=1964): 
    history = {'accuracy':[],
               'loss':[],
               'params':[]}
    
    param_values = init_params(nn_architecture, seed)
    
    for epoch in range(epochs):
        
        memory = forward_pass(X, param_values, nn_architecture)
        Y_hat = memory[-1]['A']
        history['loss'].append(get_cost_value(Y_hat, Y))
        history['accuracy'].append(get_accuracy(Y_hat, Y))
        history['params'].append(param_values)
        grad_values = back_pass(Y, memory, param_values, nn_architecture)
        param_values = update_params(grad_values, param_values, nn_architecture, learning_rate)
        
    history['params'].append(param_values)
    
    return param_values, history

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

nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

nn_architecture = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.1

X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)


make_plot(X, y, 'Data')
#X, Y, nn_architecture, epochs, learning_rate, seed = np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), nn_architecture, 100, 0.001, 1964

params, history = train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), nn_architecture, 100, 0.0001, 1964)

plt.figure()
plt.plot(history['loss'])
plt.title('loss')
plt.yscale('log')
plt.show()

plt.figure()
plt.plot(history['accuracy'])
plt.title('accuracy')
plt.show()

