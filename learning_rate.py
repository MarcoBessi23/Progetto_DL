import numpy as np
from hyperoptimizer import *
from neuralNet import *
from dataloader import load_mnist
import autograd.numpy.random as npr
from autograd import grad
from autograd.test_util import check_grads
import matplotlib.pyplot as plt
import os
from time import time

# Model parameters
layer_sizes = [784, 50, 50, 50, 10]
L2_reg = 0

# Training parameters
batch_size = 250
num_epochs = 20
meta_lr = 40 #0.001
N_data = 1000
meta_iter = 4
log_param_scale = -2.0
velocity_scale = 0.0
layer_specs = [tanh_layer(i,o) for i,o in zip(layer_sizes[:-1],layer_sizes[1:])] #[tanh_layer(784,50),tanh_layer(50,50),tanh_layer(50,10)]

N, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = train_images[:N_data, :]
train_labels = train_labels[:N_data, :]
batch_idxs = BatchList(N_data, batch_size)

#construct neural network

#num_parameters, MLP, loss = construct_nn(layer_sizes)
num_parameters, nn, loss = make_nn_funs(layer_specs, L2_reg)
npr.seed(1)
W0 = np.random.randn(num_parameters) * np.exp(log_param_scale)
V0 = np.zeros_like(W0)
log_alpha_0 = 1.0
gamma_0 = 0.9
N_iter = 5
log_alphas = np.full(N_iter, log_alpha_0)
gammas = np.full(N_iter, gamma_0)

def indexed_loss_fun(w, idxs):
    return loss(w, inputs = train_images[idxs], T = train_labels[idxs])

iteration = [i for i in range(N_iter)]
meta_iteration = [i for i in range(meta_iter)]
meta_lc = []

start = time()
for i in range(meta_iter):
    print(f'meta iter number {i}')
    res = RMD(W0, V0, indexed_loss_fun, indexed_loss_fun, gammas, log_alphas, N_iter, batch_idxs)
    hyper_g = res['hg_alpha']
    log_alphas = log_alphas - meta_lr * hyper_g
    meta_lc.append(res['loss'])
    if i == 0:
        initial_lc = res['learning curve']
    if i == meta_iter-1:
        final_lc = res['learning curve']
end = time()-start
print(f'time of HPO: {end}')

folder_path = '/home/marco/Documenti/Progetto_DL/results_learning_rate'

plt.plot(meta_iteration, meta_lc, marker='o')
plt.xlabel('meta iteration')
plt.ylabel('final loss')

meta_train = os.path.join(folder_path, "meta_training_loss.png")
plt.savefig(meta_train, dpi=300)
plt.close()

plt.plot(iteration, initial_lc, marker = 'o', color = 'blue')
plt.plot(iteration, final_lc, marker = 'o', color = 'red')
plt.xlabel('iteration')
plt.ylabel('elementary learning curve')
plt.title('initial vs final training loss')

initial_vs_final = os.path.join(folder_path, "initialVSfinal_loss.png")
plt.savefig(initial_vs_final)
plt.close()