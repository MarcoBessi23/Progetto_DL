import numpy as np
from hyperoptimizer import *
from neuralNet import *
from dataloader import load_mnist
import autograd.numpy.random as npr
from autograd import grad
from autograd.util import quick_grad_check
import matplotlib.pyplot as plt
from time import time
# Model parameters

layer_sizes = [784, 50, 50, 10]
L2_reg = 0

# Training parameters

param_scale = 0.1
batch_size = 250
num_epochs = 20
meta_step_size = 0.001
mass = 0.8
N_data = 10000
meta_iter = 4
log_param_scale = -2.0
velocity_scale = 0.0

N, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = train_images[:N_data, :]
train_labels = train_labels[:N_data, :]
batch_idxs = BatchList(N_data, batch_size)

#construct neural network

#num_parameters, MLP, loss = construct_nn(layer_sizes)
num_parameters, nn, loss = make_nn_funs(layer_specs, L2_reg)
print(num_parameters)
W0 = np.random.randn(num_parameters)
V0 = np.zeros_like(W0)
log_alpha_0 = 0.0
gamma_0 = 0.9
N_iter = 5
log_alphas = np.full(N_iter, log_alpha_0)
gammas = np.full(N_iter, gamma_0)

def indexed_loss_fun(w, idxs):
    return loss(w, inputs = train_images[idxs], T = train_labels[idxs])

from autograd.test_util import check_grads

out = []
meta_iteration = [i for i in range(meta_iter)]
print(meta_iteration)

start = time()
for i in range(meta_iter):
    print(f'meta iter number {i}')
    res = RMD(W0, V0, indexed_loss_fun, indexed_loss_fun, gammas, np.exp(log_alphas), N_iter, batch_idxs)
    hyper_g = np.exp(log_alphas) * res['hg_alpha']
    log_alphas = log_alphas + meta_step_size * hyper_g
    
    out.append(res['loss']) 

end = time()-start
print(f'time of HPO: {end}')

plt.plot(meta_iteration, out, marker='o')
plt.xlabel('meta iteration')
plt.ylabel('final loss')
plt.show()