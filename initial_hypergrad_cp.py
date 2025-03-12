import autograd.numpy as np
from hyperoptimizer import data_RMD, BatchList
from neuralNet import VectorParser, plot_images, make_nn_funs, RandomState, fill_parser
from dataloader import load_data_dicts
import autograd.numpy.random as npr
from autograd import grad
import matplotlib.pyplot as plt
from checkpointing import BinomialCKP, adjust, maxrange, ActionType, numforw


def logit(x):
    return 1 / (1 + np.exp(-x))

def inv_logit(y):
    return -np.log( 1/y - 1)

def d_logit(x):
    return logit(x) * (1 - logit(x))


# ----- Fixed params -----
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
N_meta_iter = 50
seed = 0

train_data, valid_data, tests_data = load_data_dicts(N_train, N_valid, N_tests)
parser, pred_fun, loss_fun = make_nn_funs(layer_sizes)
N_weight_types = len(parser.names)
hyperparams = VectorParser()
hyperparams['log_param_scale']  = np.full(N_weight_types, init_log_param_scale)
hyperparams['log_alphas']       = np.full((nSteps, N_weight_types), init_log_alphas)
hyperparams['invlogit_gammas']  = np.full((nSteps, N_weight_types), init_invlogit_gammas)
fixed_hyperparams = VectorParser()
fixed_hyperparams['log_L2_reg'] = np.full(N_weight_types, init_log_L2_reg)
hypergrads = VectorParser()

def f_loss(w):
    return loss_fun(w, **train_data)

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
        proj      = lambda w, d, i : np.dot(L_grad(w,i),d)
        hessianvp = grad(proj, 0)
        i, alpha, gamma = iters[iteration]

        #print(f'backprop step {i}')

        cur_alpha_vect = fill_parser(parser, alpha)
        cur_gamma_vect = fill_parser(parser, gamma)
        
        g  = L_grad(w, i)
        v_next = np.copy(v)
        v_next *= cur_gamma_vect
        v_next -= (1 - cur_gamma_vect) * g

        #print(f'valore ricostruito di v al tempo {i}')
        #print(v_next[0])

        for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.items()):
            d_alpha[i,j] = np.dot(d_w[ixs], v_next[ixs])

        ##Questi tre passaggi non devono essere fatti e i valori vanno ottenuti tramite checkpoint
        
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
            
        elif action == ActionType.firsturn:
            print('executing first reverse step')
            wF_1, vF_1 = forward(scheduler.check, scheduler.oldcapo, nSteps-1)
            wF, vF = forward(scheduler.check,scheduler.oldcapo, nSteps)

            final_loss = f(wF)
            #initialise gradient values
            d_alpha, d_gamma = np.zeros(alphas.shape), np.zeros(gammas.shape)
            d_w = f_grad(wF)
            d_v = np.zeros(d_w.shape)
            #print('valore che deve andare dentro a d_alpha')
            d_w, d_v, d_alpha, d_gamma = reverse(nSteps-1, wF_1, vF_1, d_w, d_v, d_alpha, d_gamma)
        elif action == ActionType.restore:
            #print(f'loading state number {scheduler.check}')
            w, v = np.copy(stack[scheduler.check]['weights']), np.copy(stack[scheduler.check]['velocity'])
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



initial_hypergrad = hyper_grad_lr(hyperparams.vect, 0)
hyper = np.zeros((N_meta_iter, len(initial_hypergrad)))
for i in range(N_meta_iter):
    print(f'--------META ITERATION {i}--------------')
    hyper[i] = hyper_grad_lr(hyperparams.vect, i)

avg_hypergrad        = np.mean(hyper, axis=0)
parsed_avg_hypergrad = hyperparams.new_vect(avg_hypergrad)



fig = plt.figure(0)
fig.clf()
ax = fig.add_subplot(111)

colors = ['blue', 'green', 'red', 'deepskyblue']
def layer_name(weight_key):
    return "Layer {num}".format(num=weight_key[1] + 1)
index = 0
for cur_results, name in zip(parsed_avg_hypergrad['log_alphas'].T, parser.names):
    if name[0] == 'weights':
        ax.plot(cur_results, 'o-', label=layer_name(name), color = colors[index], markeredgecolor = 'black')
        index += 1
        print(np.shape(cur_results))
#low, high = ax.get_ylim()
#ax.set_ylim([0, high])
ax.set_ylabel('Learning rate Gradient')
ax.set_xlabel('Schedule index')
ax.set_yticks([0,])
ax.set_yticklabels(['0',])
fig.set_size_inches((6,2.5))

import os
path = os.path.join(os.getcwd(), 'initial_hyper_values', 'checkpoint.png')
plt.savefig(path)

