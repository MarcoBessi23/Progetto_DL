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
from checkpointing import *


# Model parameters
layer_sizes = [784, 50, 50, 50, 10]
L2_reg = 0

# Training parameters
batch_size = 250
meta_lr = 40 #0.001
meta_mass = 0.9
N_data = 1000
meta_iter = 4
log_param_scale = -2.0
velocity_scale = 0.0
nsnaps = 50


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
W0 = np.random.randn(num_parameters) * np.exp(log_param_scale)
V0 = np.zeros_like(W0)

log_alpha_0 = 1.0
gamma_0 = 0.9
N_iter = 5
alphas = np.full(N_iter, log_alpha_0)
gammas = np.full(N_iter, gamma_0)

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


#A single meta-iteration for checkpoint 

gradient = grad(indexed_loss_fun)
nSteps = 30
multi_alpha = np.full((nSteps, len(o)), init_log_alphas)
multi_gamma = np.full((nSteps, len(o)), gamma_0)
num_epochs = int(nSteps/len(batch_idxs))+1
print('numero di epoche ricavato')
print(num_epochs)
nSnaps = adjust(nSteps)
print('numero totale di checkpoint preparati')
print(nSnaps)
scheduler = Offline(nSnaps, nSteps)        #CRevolve(nSnaps, nSteps)

"""
nfrom è un checkpoint, deve permettermi di ricreare lo stato del modello a quella iterazione quindi ci devo 
salvare dentro w e v, se per esempio ho salvato l'iterazione 5 e voglio 
arrivare a quella 10 , devo tirare fuori v[5] e w[5] da oldcapo e poi iterare sugli alpha e gamma di 
quell'intervallo alphas è una matrice, ogni riga rappresenta un'iterazione e su questa ci sono i valori 
di alpha da associare ai pesi dei vari layer, quindi alphas[0,0] è il valore di alpha che viene moltiplicato per 
i v che sottrai ai pesi del primo layer, analogo per gammas.
Quindi iters che è la lista su cui vai ad iterare dovrà contenere alphas e gammas da 5 a 10 e anche batch idxs 
"""

"""
batch idxs divide il training set in 4 batch di indici, 1000/250, all'iterazione 0 uso il primo batch
alla 1 il secondo e così via, finchè non arrivo al batch 4, a quel punto se ho più di 4 iterazioni ricomincio
dal primo batch, quindi coviene calcolarmi batches fuori e poi dentro allo zip mettere batches da iterazione 5 a 
10 nel nostro esempio(da nfrom a nto in generale)
"""

iters = list(zip(range(nSteps), multi_alpha, multi_gamma, batch_idxs * num_epochs))
print(len(iters))

#snapStack = [None]*nsnaps
snapStack = [{} for _ in range(nSnaps)] 
'''
prima devi inizializzare i valori dello stato del sistema e del gradiente della funzione che poi
vengono aggiornati
'''

l_grad = grad(indexed_loss_fun) # nel nostro caso loss function e validation loss coincidono perché il paper
                                # considera solo training data

w_0 = np.random.randn(num_parameters) * np.exp(log_param_scale)
v_0 = np.zeros_like(w_0)



def forward(nfrom: int, nto: int):

    w = snapStack[nfrom]['weights']
    v = snapStack[nfrom]['velocity']

    for t, alpha, gamma, batch in iters[nfrom:nto]:
        print(f'forward step number {t}')
        cur_alpha = load_alpha(parser, alpha)
        cur_gamma = load_alpha(parser, gamma)
        g =  gradient(w, batch)
        v *= cur_gamma
        v -= (1 - cur_gamma)*g
        w += cur_alpha*v

    return w, v

hyper_gradient = grad(lambda w, idx, d: np.dot(gradient(w,idx),d))

def reverse(iteration, w, v, d_w, d_v, d_alpha, d_gamma):
    '''
    This function does only one step of RMD
    '''
    
    i, alpha, gamma, batch = iters[iteration]
    print(f'backprop step {i}')
    cur_alpha = load_alpha(parser, alpha)
    cur_gamma = load_alpha(parser, gamma)
    for j, (ixs, _) in enumerate(parser.shape_idx.values()):
            d_alpha[i,j] = np.dot(d_w[ixs], v[ixs])

    #exact gradient descent reversion
    g  = gradient(w, batch)
    w -= cur_alpha * v
    v += (1-cur_gamma)*g
    v /= cur_gamma

    d_v += cur_alpha*d_w
    for j, (ixs, _) in enumerate(parser.shape_idx.values()):
            d_gamma[i,j] = np.dot(d_v[ixs], v[ixs] + g[ixs])
    d_w -= (1-cur_gamma) * hyper_gradient(w, batch, d_v)
    d_v *= cur_gamma

    return d_w, d_v, d_alpha, d_gamma


#w, v = w_0, v_0
#while(True):
#    action = scheduler.revolve()
#    print(action)
#    if action == ActionType.advance:
#        print(f'advance the system from {scheduler.old_capo} to {scheduler.capo}')
#        w, v = forward(scheduler.old_capo, scheduler.capo)
#    elif action == ActionType.takeshot:
#        print('saving current state')
#        print(scheduler.check)
#        snapStack[scheduler.check]['weights']  = w
#        snapStack[scheduler.check]['velocity'] = v
#    elif action == ActionType.firsturn:
#        print('executing first reverse step')
#        wF, vF = forward(scheduler.old_capo, nSteps)
#        #initialise gradient values
#        d_alpha, d_gamma = np.zeros(multi_alpha.shape), np.zeros(multi_gamma.shape)
#        d_v = np.zeros_like(w_0)
#        d_w = l_grad(wF, batch_idxs.all_idxs)  
#        #first step
#        d_w, d_v, d_alpha, d_gamma = reverse(nSteps-1, wF, vF, d_w, d_v, d_alpha, d_gamma)
#    elif action == ActionType.restore:
#        print(f'loading state number {scheduler.check}')
#        w, v = snapStack[scheduler.check]['weights'], snapStack[scheduler.check]['velocity']
#    elif action == ActionType.youturn:
#        print(f' doing reverse step at time {scheduler.fine}')
#        d_w, d_v, d_alpha, d_gamma = reverse(scheduler.fine, w, v, d_w, d_v, d_alpha, d_gamma)
#    if action == ActionType.terminate:
#        break
