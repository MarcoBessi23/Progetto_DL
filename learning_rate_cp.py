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

# Training parameters
batch_size     = 200
meta_step_size = 0.04 #0.04
N_data    = 10000
meta_iter = 5

N, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = train_images[:N_data, :]
train_labels = train_labels[:N_data, :]
batch_idxs   = BatchList(N_data, batch_size)

#construct neural network
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
seed        = 1
nSteps      = 100

#initial value of w0 and L2 reg
log_param_scale = np.full(num_parameters, init_log_param_scale)
log_L2_reg      = np.full(num_parameters, init_log_L2_reg)
#log_L2_reg      = np.exp(log_L2_reg)

def indexed_loss_fun(w, idxs):
    return loss(w, inputs = train_images[idxs], targets = train_labels[idxs], L2_reg = np.exp(log_L2_reg))

#initial values of multi alpha and multi gamma parameters
multi_alpha = np.full((nSteps, len(o)), init_log_alphas)
multi_gamma = np.full((nSteps, len(o)), init_invlogit_gammas)
num_epochs  = int(nSteps/len(batch_idxs))+1

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

def hyper_grad_lr(nSteps, parser, loss, f, multi_alpha, multi_gamma, w_0, v_0, batch_idxs, num_epochs ):

    scheduler = BinomialCKP(nSteps)
    stack     = [{} for _ in range(scheduler.snaps)]
    iters     = list(zip(range(nSteps), multi_alpha, multi_gamma, batch_idxs * num_epochs))
    gradient  = grad(loss)
    l_grad    = grad(f)
    #l_grad    = grad(control)
    w, v      = w_0, v_0


    def forward(check:int, nfrom: int, nto: int):

        w = stack[check]['weights']
        v = stack[check]['velocity']

        for t, alpha, gamma, batch in iters[nfrom:nto]:
            print(f'forward step number {t}')
            cur_alpha = load_alpha(parser, alpha)
            cur_gamma = load_alpha(parser, gamma)
            g =  gradient(w, batch)
            print('max di gradient')
            print(np.max(g))
            v *= cur_gamma
            v -= (1 - cur_gamma)*g
            print(' max di v')
            print(np.max(v))
            print('max di w')
            print(np.max(w))
            w += cur_alpha*v

        return w, v

    def reverse(iteration, w, v, d_w, d_v, d_alpha, d_gamma):
        '''
        This function does only one step of RMD
        '''
        hessianvp = grad(lambda w, idx, d: np.dot(gradient(w,idx),d))
        i, alpha, gamma, batch = iters[iteration]
        print(f'backprop step {i}')
        cur_alpha = load_alpha(parser, alpha)
        cur_gamma = load_alpha(parser, gamma)
        for j, (ixs, _) in enumerate(parser.shape_idx.values()):
                d_alpha[i,j] = np.dot(d_w[ixs], v[ixs])

        #gradient descent reversion
        g  = gradient(w, batch)
        print('max di gradient')
        print(np.max(g))
        if np.isnan(np.max(g)):
            print('NAN')
        w -= cur_alpha * v
        v += (1-cur_gamma)*g
        v /= cur_gamma
        print(' max di v')
        print(np.max(v))
        if np.isnan(np.max(v)):
            print('NAN')
        print('max di w')
        print(np.max(w))
        if np.isnan(np.max(w)):
            print('NAN')

        d_v += cur_alpha*d_w
        for j, (ixs, _) in enumerate(parser.shape_idx.values()):
                d_gamma[i,j] = np.dot(d_v[ixs], v[ixs] + g[ixs])
        d_w -= (1-cur_gamma)*hessianvp(w, batch, d_v)
        d_v *= cur_gamma
        print('dw')
        if np.isnan(np.max(d_w)):
            print('NAN')
            print(d_w)

        print('dv')
        if np.isnan(np.max(d_w)):
            print('NAN')
            print(d_v)

        return d_w, d_v, d_alpha, d_gamma

    
    while(True):
        action = scheduler.revolve()
        print(action)
        if action == ActionType.advance:
            print(f'advance the system from {scheduler.oldcapo} to {scheduler.capo}')
            w, v = forward(scheduler.check, scheduler.oldcapo, scheduler.capo)
            if np.isnan(np.max(w)):
                return 'NAN'
            if np.isnan(np.max(v)):
                return 'NAN'
        elif action == ActionType.takeshot:
            print('saving current state')
            print(scheduler.check)
            stack[scheduler.check]['weights']  = w
            stack[scheduler.check]['velocity'] = v
        elif action == ActionType.firsturn:
            print('executing first reverse step')
            wF, vF = forward(scheduler.check, scheduler.oldcapo, nSteps)
            final_loss = loss(wF, batch_idxs.all_idxs)
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
            if np.isnan(np.max(d_w)):
                return 'NAN'
            if np.isnan(np.max(d_v)):
                return 'NAN'
            if np.isnan(np.max(d_alpha)):
                return 'NAN'
            elif np.isnan(np.max(d_gamma)):
                return 'NAN'
        if action == ActionType.terminate:
            break

    return d_w, d_v, d_alpha, d_gamma, final_loss

##CAMBIARE STRUTTURA IN CUI SALVI W PER PERMETTERE AGGIORNAMENTO DI LOG_PARAM_SCALE

iteration      = [i for i in range(nSteps)]
meta_iteration = [i for i in range(meta_iter)]
meta_lc   = []
weights_1 = []
weights_2 = []
weights_3 = []
weights_4 = []
bias_1 = []
bias_2 = []
bias_3 = []
bias_4 = []


def adam_single_iter(hypergrad, hyper, iter, im, iv, step_size=0.1,
                      b1 = 0.1, b2 = 0.01, eps = 10**-4, lam=10**-4):
    '''
    im, iv sono stati inizializzati a 0, a questa funzione vengono
    passati gli iper di alpha e di gamma e aggiorna i due iper separatamente
    '''

    b1t  = 1 - (1-b1)*(lam**iter)
    im   = b1t*hypergrad     + (1-b1t)*im   # First  moment estimate
    iv   = b2*(hypergrad**2) + (1-b2)*iv    # Second moment estimate
    mhat = im/(1-(1-b1)**(iter+1))          # Bias correction
    vhat = iv/(1-(1-b2)**(iter+1))
    hyper -= step_size*mhat/(np.sqrt(vhat) + eps)

    return hyper, im, iv

ima, iva   = np.zeros_like(multi_alpha), np.zeros_like(multi_alpha)
img, ivg   = np.zeros_like(multi_gamma), np.zeros_like(multi_gamma)
imp, ivp   = np.zeros_like(log_param_scale), np.zeros_like(log_param_scale)
loss_final = []
for i in range(meta_iter):
    print(f'------------------------META ITERATION {i}--------------------------------------')
    #d_w, d_v, hyper_alpha, hyper_gamma = hyper_grad_lr(nSteps, parser, indexed_loss_fun, indexed_loss_fun, 
    #                                                   multi_alpha, multi_gamma, w_0, v_0, batch_idxs, num_epochs)    

    w_0 = np.exp(log_param_scale)
    rs = RandomState((seed, meta_iter))
    w_0 *= rs.randn(w_0.size)
    v_0 = np.zeros_like(w_0)
    res = hyper_grad_lr(nSteps, parser, indexed_loss_fun, indexed_loss_fun,
                        np.exp(multi_alpha), logit(multi_gamma), w_0, v_0, batch_idxs, num_epochs)
    print('results:')
    print(res)
    if res == 'NAN':
        print('NAN da qualche parte')
        break
    else:
        hypergrad_alpha = np.exp(multi_alpha) * res[2] #derivate rispetto ad hyper
        hypergrad_gamma = d_logit(multi_gamma)* res[3] #derivate rispetto ad hyper
        hypergrad_param_scale = w_0 * res[0]

        loss_final.append(res[4])
        print('hypergrad wrt alpha')
        print(hypergrad_alpha)
        #lavori nello spazio logaritmico
        multi_gamma, ima, iva = adam_single_iter(hypergrad_gamma, multi_gamma, i, img, ivg, step_size= meta_step_size)
        multi_alpha, img, ivg = adam_single_iter(hypergrad_alpha, multi_alpha, i, ima, iva, step_size= meta_step_size)
        log_param_scale, imp, ivp = adam_single_iter(hypergrad_param_scale, log_param_scale, i, imp, ivp, step_size= meta_step_size)

#riporti i valori fuori dallo spazio logaritmico
multi_alpha = np.exp(multi_alpha)
multi_gamma = logit(multi_gamma)

label = 'Layer 1'
#print(label)
#print(multi_alpha[:, 0])
plt.plot(iteration, multi_alpha[:, 0], marker = 'o', label = label, color = 'blue')

label = 'Layer 2'
#print(label)
#print(multi_alpha[:, 2])
plt.plot(iteration, multi_alpha[:, 2], marker = 'o', label = label, color = 'green')

label = 'Layer 3'
#print(label)
#print(multi_alpha[:, 4])
plt.plot(iteration, multi_alpha[:, 4], marker = 'o', label = label, color = 'red')

label = 'Layer 4'
#print(label)
#print(multi_alpha[:, 6])
folder_path = '/home/marco/Documenti/Progetto_DL/results_learning_rate'
learning_rate_schedule = os.path.join(folder_path, "learning_schedule_cp.png")
plt.plot(iteration, multi_alpha[:, 6], marker = 'o', label = label, color = 'yellow')
plt.xlabel('iteration')
plt.ylabel('learning rate')
plt.savefig(learning_rate_schedule, dpi=300)
plt.close()
plt.plot(meta_iteration, loss_final)
plt.xlabel('meta iteration')
plt.ylabel('loss')
plt.show()
plt.close()