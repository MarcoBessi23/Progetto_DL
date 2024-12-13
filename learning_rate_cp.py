import numpy as np
from hyperoptimizer import *
from neuralNet import *
from dataloader import load_mnist
import autograd.numpy.random as npr
from autograd import grad
from autograd.test_util import check_grads
import matplotlib.pyplot as plt
from time import time
import os

from checkpointing import Checkpoint, BinomialCKP, adjust, maxrange, ActionType, numforw  #hyper_grad_lr


# Model parameters
layer_sizes = [784, 50, 50, 50, 10]
L2_reg = 0

# Training parameters
batch_size = 250
meta_step_size = 0.04 #0.001
meta_mass = 0.9
N_data = 10000
meta_iter = 5
log_param_scale = -2.0


N, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = train_images[:N_data, :]
train_labels = train_labels[:N_data, :]
batch_idxs = BatchList(N_data, batch_size)

#construct neural network
parser, MLP, loss = construct_nn(layer_sizes)
npr.seed(1)
o = parser.shape_idx
print(len(o)) # number of hyper
init_log_alphas = 1.0

num_parameters = parser.N
log_alpha_0 = 1.0
gamma_0 = 0.9
N_iter = 100

def indexed_loss_fun(w, idxs):
    return loss(w, inputs = train_images[idxs], targets = train_labels[idxs], L2_reg = L2_reg)

iteration = [i for i in range(N_iter)]
meta_iteration = [i for i in range(meta_iter)]
meta_lc = []
weights_1 = []
weights_2 = []
weights_3 = []
weights_4 = []
bias_1 = []
bias_2 = []
bias_3 = []
bias_4 = []

nSteps = 100
multi_alpha = np.full((nSteps, len(o)), init_log_alphas)
multi_gamma = np.full((nSteps, len(o)), gamma_0)
num_epochs = int(nSteps/len(batch_idxs))+1

"""
nfrom is a checkpoint; it must allow me to recreate the state of the model at that iteration. 
Therefore, I need to save w and v in it. For example, if I saved iteration 5 and want to reach
iteration 10, I need to retrieve v[5] and w[5] from oldcapo and then iterate over the alpha and
gamma values for that range.

alphas is a matrix where each row represents an iteration, and the row contains the alpha values
associated with the weights of the various layers. For instance, alphas[0,0] is the alpha value 
that is multiplied by the v values, which are then subtracted from the weights of the first layer. 
The same applies for gammas.

Thus, iters, which is the list you iterate over, must include alphas and gammas from 5 to 10, as well as 
the corresponding batch indices (batch_idxs).

batch_idxs splits the training(assuming N_data = 1000) set into 4 batches of indices (1000/250). At iteration 0, 
I use the first batch, at iteration 1 the second, and so on, until I reach batch 4. At that point, 
if there are more than 4 iterations, I start again from the first batch.
Therefore, it’s convenient to calculate the batches beforehand and then, inside the zip, 
include the batches from iteration 5 to 10 in our example (from nfrom to nto in general).
"""

'''
First, you need to initialize the values of the system state and the function gradient, which are then updated.
'''
num_parameters = parser.N
w_0 = np.random.randn(num_parameters) * np.exp(log_param_scale)
v_0 = np.zeros_like(w_0)


def hyper_grad_lr(nSteps, parser, loss, f, multi_alpha, multi_gamma, w_0, v_0, batch_idxs, num_epochs ):

    scheduler = BinomialCKP(nSteps)
    stack = [{} for _ in range(scheduler.snaps)]
    iters = list(zip(range(nSteps), multi_alpha, multi_gamma, batch_idxs * num_epochs))
    gradient = grad(loss)
    l_grad   = grad(f)
    w, v = w_0, v_0


    def forward(check:int, nfrom: int, nto: int):

        w = stack[check]['weights']
        v = stack[check]['velocity']

        for t, alpha, gamma, batch in iters[nfrom:nto]:
            print(f'forward step number {t}')
            cur_alpha = load_alpha(parser, alpha)
            cur_gamma = load_alpha(parser, gamma)
            g =  gradient(w, batch)
            v *= cur_gamma
            v -= (1 - cur_gamma)*g
            w += cur_alpha*v

        return w, v

    def reverse(iteration, w, v, d_w, d_v, d_alpha, d_gamma):
        '''
        This function does only one step of RMD
        '''
        hyper_gradient = grad(lambda w, idx, d: np.dot(gradient(w,idx),d))
        i, alpha, gamma, batch = iters[iteration]
        print(f'backprop step {i}')
        cur_alpha = load_alpha(parser, alpha)
        cur_gamma = load_alpha(parser, gamma)
        for j, (ixs, _) in enumerate(parser.shape_idx.values()):
                d_alpha[i,j] = np.dot(d_w[ixs], v[ixs])

        #gradient descent reversion
        g  = gradient(w, batch)
        w -= cur_alpha * v
        v += (1-cur_gamma)*g
        v /= cur_gamma

        d_v += cur_alpha*d_w
        for j, (ixs, _) in enumerate(parser.shape_idx.values()):
                d_gamma[i,j] = np.dot(d_v[ixs], v[ixs] + g[ixs])
        print('get dw')
        d_w -= (1-cur_gamma)*hyper_gradient(w, batch, d_v)
        print('done')
        d_v *= cur_gamma

        return d_w, d_v, d_alpha, d_gamma

    
    while(True):
        action = scheduler.revolve()
        print(action)
        if action == ActionType.advance:
            print(f'advance the system from {scheduler.oldcapo} to {scheduler.capo}')
            w, v = forward(scheduler.check, scheduler.oldcapo, scheduler.capo)
        elif action == ActionType.takeshot:
            print('saving current state')
            print(scheduler.check)
            stack[scheduler.check]['weights']  = w
            stack[scheduler.check]['velocity'] = v
        elif action == ActionType.firsturn:
            print('executing first reverse step')
            wF, vF = forward(scheduler.check, scheduler.oldcapo, nSteps)
            #initialise gradient values
            d_alpha, d_gamma = np.zeros(multi_alpha.shape), np.zeros(multi_gamma.shape)
            d_v = np.zeros_like(w_0)
            d_w = l_grad(wF, batch_idxs.all_idxs)  
            #first step
            d_w, d_v, d_alpha, d_gamma = reverse(nSteps-1, wF, vF, d_w, d_v, d_alpha, d_gamma)
        elif action == ActionType.restore:
            print(f'loading state number {scheduler.check}')
            w, v = stack[scheduler.check]['weights'], stack[scheduler.check]['velocity']
        elif action == ActionType.youturn:
            print(f' doing reverse step at time {scheduler.fine}')
            d_w, d_v, d_alpha, d_gamma = reverse(scheduler.fine, w, v, d_w, d_v, d_alpha, d_gamma)
        if action == ActionType.terminate:
            break

    return d_w, d_v, d_alpha, d_gamma


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
    d_w, d_v, hyper_alpha, hyper_gamma = hyper_grad_lr(nSteps, parser, indexed_loss_fun, indexed_loss_fun, multi_alpha
                        ,multi_gamma, w_0, v_0, batch_idxs, num_epochs)

    hyper_lr = hyper_alpha
    hyper_momentum = hyper_gamma
    multi_gamma = multi_gamma + meta_mass * hyper_momentum
    multi_alpha = multi_alpha + meta_step_size * hyper_lr

    if i == meta_iter-1:
        grad_vec = hyper_lr

for i in range(len(layer_sizes)-1):
    plt.plot(iteration,multi_alpha[:, 2*i], marker = 'o')
plt.xlabel('iteration')
plt.ylabel('learning rate')
plt.show()
plt.close()

plt.plot(iteration, grad_vec[:,0], color= 'red', marker = 'o')
plt.plot(iteration, grad_vec[:,2], color= 'yellow', marker = 'o')
plt.plot(iteration, grad_vec[:,4], color= 'blue', marker = 'o')
plt.plot(iteration, grad_vec[:,6], color= 'green', marker = 'o')
plt.xlabel('iteration')
plt.ylabel('hypergradient')
plt.show()

plt.plot(meta_iteration, weights_1, color= 'red', marker = 'o')
plt.plot(meta_iteration, weights_2, color= 'yellow', marker = 'o')
plt.plot(meta_iteration, weights_3, color= 'blue', marker = 'o')
plt.plot(meta_iteration, weights_4, color= 'green', marker = 'o')
plt.xlabel('meta iteration')
plt.ylabel('weights scale')
plt.show()

plt.close()


plt.plot(meta_iteration, bias_1, color= 'red', marker = 'o')
plt.plot(meta_iteration, bias_2, color= 'yellow', marker = 'o')
plt.plot(meta_iteration, bias_3, color= 'blue', marker = 'o')
plt.plot(meta_iteration, bias_4, color= 'green', marker = 'o')
plt.xlabel('meta iteration')
plt.ylabel('bias scale')

plt.show()
