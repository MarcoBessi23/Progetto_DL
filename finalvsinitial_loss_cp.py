import numpy as np
from hyperoptimizer import *
from neuralNet import *
from dataloader import load_data_dicts 
import autograd.numpy.random as npr
from autograd import grad
from autograd.test_util import check_grads
import matplotlib.pyplot as plt
from time import time
import os
from checkpointing import Checkpoint, BinomialCKP, adjust, maxrange, ActionType, numforw  #hyper_grad_lr

def logit(x):
    return 1 / (1 + np.exp(-x))

def inv_logit(y):
    return -np.log( 1/y - 1)

def d_logit(x):
    return logit(x) * (1 - logit(x))


layer_sizes = [784, 50, 50, 50, 10]
batch_size = 200
nSteps = 100
N_classes = 10
N_train = 10000
N_valid = 10000
N_tests = 10000

# ----- Initial values of learned hyper-parameters -----
init_log_L2_reg = -100.0
init_log_alphas = -1.0
init_invlogit_gammas = inv_logit(0.5)
init_log_param_scale = -3.0

# ----- Superparameters -----
meta_alpha = 0.04
N_meta_iter = 50

seed = 0
train_data, valid_data, tests_data = load_data_dicts(N_train, N_valid, N_tests)
parser, pred_fun, loss_fun         = make_nn_funs(layer_sizes)
N_weight_types = len(parser.names)


#iper che vengono aggiornati
hyperparams = VectorParser()
hyperparams['log_param_scale']  = np.full(N_weight_types, init_log_param_scale)
hyperparams['log_alphas']       = np.full((nSteps, N_weight_types), init_log_alphas)
hyperparams['invlogit_gammas']  = np.full((nSteps, N_weight_types), init_invlogit_gammas)
fixed_hyperparams = VectorParser()
fixed_hyperparams['log_L2_reg'] = np.full(N_weight_types, init_log_L2_reg)
hypergrads = VectorParser()
L2_reg = fill_parser(parser, np.exp(fixed_hyperparams['log_L2_reg']))


#def indexed_loss_fun(w, idxs):
#    return loss_fun(w,  X = train_data['X'][idxs], T = train_data['T'][idxs], L2_reg = L2_reg)

learning_curves = {}
loss_final = []
def f_loss(w):
    return loss_fun(w, **train_data)

def training_set(i_hyper):
    idx_set = set()
    for i in range(100):
        rs = RandomState((seed, i_hyper, i))
        batch = rs.randint(N_train, size=batch_size)
        idx_set.update(batch)
    
    return list(idx_set)

def hyper_grad_lr(hyperparam_vec, i_hyper):

    cur_hyper = hyperparams.new_vect(hyperparam_vec)
    rs        = RandomState((seed, i_hyper))
    w_0       = fill_parser(parser, np.exp(cur_hyper['log_param_scale']))
    w_0      *= rs.randn(w_0.size)
    v_0       = np.zeros_like(w_0)
    L2_reg    = fill_parser(parser, np.exp(fixed_hyperparams['log_L2_reg']))

    def indexed_loss_fun(w, i_iter):
        rs = RandomState((seed, i_hyper, i_iter))  # Deterministic seed needed for backwards pass.
        idxs = rs.randint(N_train, size=batch_size)
        return loss_fun(w, train_data['X'][idxs], train_data['T'][idxs], L2_reg)

    loss = indexed_loss_fun
    f = f_loss


    scheduler = BinomialCKP(nSteps)
    stack     = [{} for _ in range(scheduler.snaps)]
    alphas    = np.exp(cur_hyper['log_alphas'])
    gammas    = logit(cur_hyper['invlogit_gammas'])
    iters     = list(zip(range(len(alphas)), alphas, gammas))
    L_grad    = grad(loss)
    f_grad    = grad(f)
    w, v      = np.copy(w_0), np.copy(v_0)
    

    def record_learning_curve():
        wt, vt = np.copy(w_0), np.copy(v_0)
        learning_curve = []
        learning_curve.append(f(wt))
        for i, alpha, gamma in iters:

            g = L_grad(wt, i)
            cur_alpha_vect = fill_parser(parser, alpha)
            cur_gamma_vect = fill_parser(parser, gamma)

            vt *= cur_gamma_vect
            vt -= (1 - cur_gamma_vect) * g
            wt += cur_alpha_vect * vt
            #print(f'v al forward step {i}')
            #print(v[0:4])
            learning_curve.append(f(wt))

        return learning_curve

    if i_hyper == 1:
        learning_curves['first_loss']= record_learning_curve() 
    if i_hyper == N_meta_iter-1:
        learning_curves['final_loss']= record_learning_curve()

    def forward(check:int, nfrom: int, nto: int):

        w = np.copy(stack[check]['weights'])
        v = np.copy(stack[check]['velocity'])
        for i, alpha, gamma in iters[nfrom:nto]:

            g = L_grad(w, i)
            cur_alpha_vect = fill_parser(parser, alpha)
            cur_gamma_vect = fill_parser(parser, gamma)

            v *= cur_gamma_vect
            v -= (1 - cur_gamma_vect) * g
            w += cur_alpha_vect * v
        
        return w, v

    def reverse(iteration, w, v, d_w, d_v, d_alpha, d_gamma):
        '''
        This function does only one step of RMD
        riceve w e v ma non deve invertirli tramite gamma, poi fa un passo di forward
        per aggiornare v e poterlo mettere in d_alpha 
        w e v che servono a questa funzione sono quelli che RMD calcola invertendo il gradiente in maniera
        esatta.
        '''
        proj = lambda w, d, i : np.dot(L_grad(w,i),d)
        hessianvp = grad(proj, 0)
        i, alpha, gamma = iters[iteration]

        print(f'backprop step {i}')
        
        cur_alpha_vect = fill_parser(parser, alpha)
        cur_gamma_vect = fill_parser(parser, gamma)
        
        g  = L_grad(w, i)
        v_next = np.copy(v)
        v_next *= cur_gamma_vect
        v_next -= (1 - cur_gamma_vect) * g

        
        for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.items()):
            d_alpha[i,j] = np.dot(d_w[ixs], v_next[ixs])
        
        d_v += d_w * cur_alpha_vect

        for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.items()):
                d_gamma[i,j] = np.dot(d_v[ixs], v[ixs] + g[ixs])

        d_w -= hessianvp(w, (1-cur_gamma_vect)*d_v, i)
        d_v *= cur_gamma_vect

        return d_w, d_v, d_alpha, d_gamma

    while(True):
        action = scheduler.revolve()
        #print(action)
        if action == ActionType.advance:
            #print(f'advance the system from {scheduler.oldcapo} to {scheduler.capo}')
            w, v = forward(scheduler.check, scheduler.oldcapo, scheduler.capo)
        elif action == ActionType.takeshot:
            #print('saving current state')
            #print(scheduler.check)
            stack[scheduler.check]['weights']  = np.copy(w)
            stack[scheduler.check]['velocity'] = np.copy(v)
            
            #print(v[0:4])
        elif action == ActionType.firsturn:

            print('executing first reverse step')
            wF_1, vF_1 = forward(scheduler.check, scheduler.oldcapo, nSteps-1)
            wF, vF = forward(scheduler.check,scheduler.oldcapo, nSteps)
            final_loss = f(wF)
            loss_final.append(final_loss)
            #initialise gradient values
            d_alpha, d_gamma = np.zeros(alphas.shape), np.zeros(gammas.shape)
            d_w = f_grad(wF)
            d_v = np.zeros(d_w.shape)
            #print('valore che deve andare dentro a d_alpha')
            d_w, d_v, d_alpha, d_gamma = reverse(nSteps-1, wF_1, vF_1, d_w, d_v, d_alpha, d_gamma)
        elif action == ActionType.restore:
            #print(f'loading state number {scheduler.check}')
            w, v = np.copy(stack[scheduler.check]['weights']), np.copy(stack[scheduler.check]['velocity'])
            #print('valore di v che viene richiamato con restore')
            #print(v[0:4])
        elif action == ActionType.youturn:
            #print(f' doing reverse step at time {scheduler.fine}')
            d_w, d_v, d_alpha, d_gamma = reverse(scheduler.fine, w, v, d_w, d_v, d_alpha, d_gamma)
        if action == ActionType.terminate: 
            break

    weights_grad = parser.new_vect(w_0 * d_w)
    hypergrads['log_param_scale'] = [np.sum(weights_grad[name])
                                     for name in weights_grad.names]
    hypergrads['log_alphas']      =  d_alpha * alphas
    hypergrads['invlogit_gammas'] = (d_gamma * d_logit(cur_hyper['invlogit_gammas']))
    
    return hypergrads.vect           

#def record_learning_curve(hyperparam_vec, i_hyper):
#        learning_curve = []
#        cur_hyper = hyperparams.new_vect(hyperparam_vec)
#        rs        = RandomState((seed, i_hyper))
#        w_0       = fill_parser(parser, np.exp(cur_hyper['log_param_scale']))
#        w_0      *= rs.randn(w_0.size)
#        v_0       = np.zeros_like(w_0)
#        L2_reg    = fill_parser(parser, np.exp(fixed_hyperparams['log_L2_reg']))
#        loss = indexed_loss_fun
#        f = f_loss
#
#        alphas    = np.exp(cur_hyper['log_alphas'])
#        gammas    = logit(cur_hyper['invlogit_gammas'])
#        iters     = list(zip(range(len(alphas)), alphas, gammas))
#        L_grad    = grad(loss)
#        f_grad    = grad(f)
#        wt, vt = np.copy(w_0), np.copy(v_0)
#        learning_curve.append(f(wt))
#        for i, alpha, gamma in iters:
#
#            g = L_grad(wt, i)
#            cur_alpha_vect = fill_parser(parser, alpha)
#            cur_gamma_vect = fill_parser(parser, gamma)
#
#            vt *= cur_gamma_vect
#            vt -= (1 - cur_gamma_vect) * g
#            wt += cur_alpha_vect * vt
#            #print(f'v al forward step {i}')
#            #print(v[0:4])
#            learning_curve.append(f(wt))
#
#        return learning_curve


#initial_result = hyper_adam(hyper_grad_lr, hyperparams.vect, N_meta_iter, meta_alpha)
#initial_hyper  = hyperparams.new_vect(initial_result)


final_result = hyper_adam(hyper_grad_lr, hyperparams.vect, N_meta_iter, meta_alpha)
final_hyper  = hyperparams.new_vect(final_result)



plt.plot(learning_curves['first_loss'], color = 'blue')
plt.plot(learning_curves['final_loss'], color = 'green')
plt.savefig('/home/marco/Documenti/Progetto_DL/results_learning_rate/initialvsfinal_cp.png')
