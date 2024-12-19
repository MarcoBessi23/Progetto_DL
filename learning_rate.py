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
# Training parameters

batch_size = 200
num_epochs = 20
meta_lr    = 0.04
N_data     = 10000
meta_iter  = 4

N, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = train_images[:N_data, :]
train_labels = train_labels[:N_data, :]
batch_idxs   = BatchList(N_data, batch_size)

#construct neural network

N_iter = 100
parser, MLP, loss = construct_nn_multi(layer_sizes)
o = parser.shape_idx

def logit(x): 
    return 1 / (1 + np.exp(-x))

def inv_logit(y): 
    return -np.log( 1/y - 1)

def d_logit(x): 
    return logit(x) * (1 - logit(x))

init_log_L2_reg = -100.0
init_log_alphas = -1.0
init_invlogit_gammas = inv_logit(0.5)
init_log_param_scale = -3.0
num_parameters  = parser.N
seed            = 1
log_param_scale = np.full(len(o), init_log_param_scale)
log_L2_reg      = np.full(num_parameters, init_log_L2_reg)
multi_alpha     = np.full((N_iter, len(o)), init_log_alphas)
multi_gamma     = np.full((N_iter, len(o)), -init_invlogit_gammas)
num_epochs      = int(N_iter/len(batch_idxs))+1


def indexed_loss_fun(w, idxs):
    return loss(w, inputs = train_images[idxs], targets = train_labels[idxs], L2_reg = np.exp(log_L2_reg))

iteration      = [i for i in range(N_iter)]
meta_iteration = [i for i in range(meta_iter)]
meta_lc   = []
weights_1 = []
weights_2 = []
weights_3 = []
weights_4 = []
bias_1    = []
bias_2    = []
bias_3    = []
bias_4    = []

meta_iter = 1
print(multi_gamma)
for i in range(meta_iter):
    print(f'meta iter {i}')
    weights_1.append(multi_alpha[0,0])
    bias_1.append(multi_alpha[0,1])
    weights_2.append(multi_alpha[0,2])
    bias_2.append(multi_alpha[0,3])
    weights_3.append(multi_alpha[0,4])
    bias_3.append(multi_alpha[0,5])
    weights_4.append(multi_alpha[0,6])
    bias_4.append(multi_alpha[0,7])

    W0  = np.exp(load_alpha(parser, log_param_scale))
    rs  = RandomState((seed, meta_iter))
    W0 *= rs.randn(W0.size)
    V0  = np.zeros_like(W0)
    res = multi_RMD(W0, V0, parser, indexed_loss_fun, indexed_loss_fun, logit(multi_gamma), np.exp(multi_alpha), 
                    N_iter, batch_idxs, True)
    print(res)
    #hyper_lr       = res['hg_alpha']
    #hyper_momentum = res['hg_gamma']
    #multi_gamma    = multi_gamma + meta_lr * hyper_momentum
    #multi_alpha    = multi_alpha + meta_lr * hyper_lr
#
    #if i == meta_iter-1:
    #    grad_vec = hyper_lr

#print(res['loss'])

#for i in range(len(layer_sizes)-1):
#    plt.plot(iteration,multi_alpha[:, 2*i], marker = 'o')
#plt.xlabel('iteration')
#plt.ylabel('learning rate')
#plt.show()
#plt.close()

#plt.plot(iteration, grad_vec[:,0], color= 'red', marker = 'o')
#plt.plot(iteration, grad_vec[:,2], color= 'yellow', marker = 'o')
#plt.plot(iteration, grad_vec[:,4], color= 'blue', marker = 'o')
#plt.plot(iteration, grad_vec[:,6], color= 'green', marker = 'o')
#plt.xlabel('iteration')
#plt.ylabel('hypergradient')
#plt.show()
#
#plt.plot(meta_iteration, weights_1, color= 'red', marker = 'o')
#plt.plot(meta_iteration, weights_2, color= 'yellow', marker = 'o')
#plt.plot(meta_iteration, weights_3, color= 'blue', marker = 'o')
#plt.plot(meta_iteration, weights_4, color= 'green', marker = 'o')
#plt.xlabel('meta iteration')
#plt.ylabel('weights scale')
#plt.show()
#
#plt.close()
#
#plt.plot(meta_iteration, bias_1, color= 'red', marker = 'o')
#plt.plot(meta_iteration, bias_2, color= 'yellow', marker = 'o')
#plt.plot(meta_iteration, bias_3, color= 'blue', marker = 'o')
#plt.plot(meta_iteration, bias_4, color= 'green', marker = 'o')
#plt.xlabel('meta iteration')
#plt.ylabel('bias scale')
#
#plt.show()
#
#start = time()
#for i in range(meta_iter):
#    print(f'meta iter number {i}')
#    res = RMD(W0, V0, indexed_loss_fun, indexed_loss_fun, gammas, log_alphas, N_iter, batch_idxs)
#    hyper_g = res['hg_alpha']
#    log_alphas = log_alphas - meta_lr * hyper_g
#    meta_lc.append(res['loss'])
#    if i == 0:
#        initial_lc = res['learning curve']
#    if i == meta_iter-1:
#        final_lc = res['learning curve']
#end = time()-start
#print(f'time of HPO: {end}')
#
#folder_path = '/home/marco/Documenti/Progetto_DL/results_learning_rate'
#
#plt.plot(meta_iteration, meta_lc, marker='o')
#plt.xlabel('meta iteration')
#plt.ylabel('final loss')
#
#meta_train = os.path.join(folder_path, "meta_training_loss.png")
#plt.savefig(meta_train, dpi=300)
#plt.close()
#
#plt.plot(iteration, initial_lc, marker = 'o', color = 'blue')
#plt.plot(iteration, final_lc, marker = 'o', color = 'red')
#plt.xlabel('iteration')
#plt.ylabel('elementary learning curve')
#plt.title('initial vs final training loss')
#
#initial_vs_final = os.path.join(folder_path, "initialVSfinal_loss.png")
#plt.savefig(initial_vs_final)
#plt.close()