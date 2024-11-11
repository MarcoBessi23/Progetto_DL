import numpy as np
from hyperoptimizer import *
from neuralNet import *
from dataloader import load_mnist
import autograd.numpy.random as npr
from autograd import grad
import matplotlib.pyplot as plt
from time import time


layer_sizes = [784,10]
L2_reg = np.random.randn(7850)
batch_size = 250
num_epochs = 20
meta_lr = 40 #0.001
meta_mass = 0.9
N_data = 1000
meta_iter = 4
N, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = train_images[:N_data, :]
train_labels = train_labels[:N_data, :]
batch_idxs = BatchList(N_data, batch_size)

parser, nn, loss = construct_nn_reg(layer_sizes)
num_parameters = parser.N
W0 = np.random.randn(num_parameters)
out = nn(W0, train_images[25,:])
print(np.shape(out))

l_grad = grad(loss)
d_w = l_grad(W0, L2_reg, train_images[25,:], train_labels[25,:] )
print(np.shape(d_w))
fun = lambda w, L2, t, l, d: np.dot(l_grad(w,L2,t,l),d)
hyper_gradient = grad(fun, 0)
hyper_theta = grad(fun, 1)

d_v = np.zeros_like(W0)
d_w = hyper_gradient(W0, L2_reg, train_images[25,:], train_labels[25,:], d_v)
d_theta = hyper_theta(W0, L2_reg, train_images[25,:], train_labels[25,:], d_v)
print(np.shape(d_w))
print(np.shape(d_theta))

#for i in range (meta_iter):
#    res = L2_RMD()
#    L2_reg = L2_reg - L2_step * hyper 

