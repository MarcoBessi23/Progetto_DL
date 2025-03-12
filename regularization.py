import numpy as np
from hyperoptimizer import *
from neuralNet import *
from dataloader import load_data_subset
import autograd.numpy.random as npr
from autograd import grad
import matplotlib.pyplot as plt
from time import time


# Not going to learn:
velocity_scale = 0.0
log_alpha_0 = 0.0
gamma_0 = 0.9
log_param_scale = -4
log_L2_reg_scale = np.log(0.01)

# ----- Discrete training hyper-parameters -----
layer_sizes = [784, 10]
batch_size  = 200
N_iters = 50

# ----- Variables for meta-optimization -----
N_train_data = 10000
N_val_data = 10000
N_test_data = 1000
meta_stepsize = 1000
N_meta_iter = 50
meta_L2_reg = 0.01

one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data_subset(N_train_data, N_val_data, N_test_data)

batch_idxs = BatchList(N_train_data, batch_size)
parser, _, loss_fun = construct_nn(layer_sizes)
N_weights = parser.N

hyperparser = WeightsParser()
hyperparser.add_weights('log_L2_reg', (N_weights,))
metas = np.zeros(hyperparser.N)

npr.seed(0)
hyperparser.set(metas, 'log_L2_reg', log_L2_reg_scale + np.ones(N_weights))

def indexed_loss_fun(x, meta_params, idxs):   # To be optimized by SGD.
    L2_reg=np.exp(hyperparser.get(meta_params, 'log_L2_reg'))
    return loss_fun(x, X=train_images[idxs], T=train_labels[idxs], L2_reg=L2_reg)

def meta_loss_fun(x, meta_params):            # To be optimized in the outer loop.
    L2_reg=np.exp(hyperparser.get(meta_params, 'log_L2_reg'))
    log_prior = -meta_L2_reg * np.dot(L2_reg.ravel(), L2_reg.ravel())
    return loss_fun(x, X=val_images, T=val_labels) - log_prior

def test_loss_fun(x):                         # To measure actual performance.
    return loss_fun(x, X=test_images, T=test_labels)

log_alphas  = np.full(N_iters, log_alpha_0)
gammas      = np.full(N_iters, gamma_0)

v0 = npr.randn(N_weights) * velocity_scale
w0 = npr.randn(N_weights) * np.exp(log_param_scale)

output = []
for i in range(N_meta_iter):
    print(f'---------------META ITERATION {i}----------------------------------')
    results = L2_RMD(indexed_loss_fun, meta_loss_fun,  N_iters, batch_idxs,
                     w0, v0, gammas, np.exp(log_alphas), metas)

    learning_curve = results['learning_curve']
    validation_loss = results['M_final']
    test_loss = test_loss_fun(results['w_final'])
    output.append((learning_curve, validation_loss, test_loss,
                   parser.get(results['w_final'], (('weights', 0))),
                   parser.get(np.exp(hyperparser.get(metas, 'log_L2_reg')), (('weights', 0)))))
    metas -= results['hM_meta'] * meta_stepsize
    



fig = plt.figure(0)
plt.clf()
ax = plt.Axes(fig, [0., 0., 1., 1.])
fig.add_axes(ax)
all_L2 = output[-1][-1]
print(np.shape(all_L2))
images = all_L2.T.copy()   
print('tipo e forma di images:')
print(type(images))
print(np.shape(images))
newmax = np.percentile(images.ravel(), 98.0)
over_ixs = images > newmax
images[over_ixs] = newmax

cax  = plot_images(images, ax, ims_per_row=5, padding=2, vmin=0.0)
cbar = fig.colorbar(cax, ticks=[0, newmax], shrink=.7)
cbar.ax.set_yticklabels(['0', '{:2.2f}'.format(newmax)])
import os
path_reg = os.path.join(os.getcwd(), 'regularization', 'penalties.png')
plt.savefig(path_reg)
