import numpy as np
from hyperoptimizer import *
from neuralNet import *
from dataloader import load_data_dicts
import autograd.numpy.random as npr
from autograd import grad
from autograd.test_util import check_grads
import matplotlib.pyplot as plt
import os
from time import time

def logit(x):
    return 1 / (1 + np.exp(-x))

def inv_logit(y):
    return -np.log( 1/y - 1)

def d_logit(x):
    return logit(x) * (1 - logit(x))


layer_sizes = [784, 50, 50, 50, 10]
batch_size  = 200
N_iters   = 100
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
parser, pred_fun, loss_fun = make_nn_funs(layer_sizes)
N_weight_types = len(parser.names)
hyperparams = VectorParser()
hyperparams['log_param_scale'] = np.full(N_weight_types, init_log_param_scale)
hyperparams['log_alphas']      = np.full((N_iters, N_weight_types), init_log_alphas)
hyperparams['invlogit_gammas']  = np.full((N_iters, N_weight_types), init_invlogit_gammas)
fixed_hyperparams = VectorParser()
fixed_hyperparams['log_L2_reg'] = np.full(N_weight_types, init_log_L2_reg)
hypergrads = VectorParser()

loss_final = []
def f_loss(w):
    return loss_fun(w, **train_data)


list = []
def hyper_gradient(hyperparams_vec, i_hyper):
    '''
    This function takes the hyperparameter vector, the meta iteration and outputs the hypergradient 
    '''
    cur_hyperparams = hyperparams.new_vect(hyperparams_vec)
    list.append(cur_hyperparams['log_param_scale'].copy())
    print(cur_hyperparams['log_param_scale'])
    rs = RandomState((seed, i_hyper))
    W0 = fill_parser(parser, np.exp(cur_hyperparams['log_param_scale']))
    W0 *= rs.randn(W0.size)
    alphas  = np.exp(cur_hyperparams['log_alphas'])
    gammas  = logit(cur_hyperparams['invlogit_gammas'])
    L2_reg = fill_parser(parser, np.exp(fixed_hyperparams['log_L2_reg']))

    def indexed_loss_fun(w, i_iter):
        rs = RandomState((seed, i_hyper, i_iter))  # Deterministic seed needed for backwards pass.
        idxs = rs.randint(N_train, size=batch_size)
        return loss_fun(w, train_data['X'][idxs], train_data['T'][idxs], L2_reg)

    hyper_list = [W0, alphas, gammas]
    res = RMD_parsed(parser, hyper_list, indexed_loss_fun, f_loss) # al posto di training_set e all_idxs
    weights_grad = parser.new_vect(W0 * res[0])
    hypergrads['log_param_scale'] = [np.sum(weights_grad[name])
                                     for name in weights_grad.names]
    hypergrads['log_alphas']      =  res[1] * alphas
    hypergrads['invlogit_gammas'] = (res[2] * d_logit(cur_hyperparams['invlogit_gammas']))
    loss_final.append(f_loss(res[3]))


    return hypergrads.vect


final_result = hyper_adam(hyper_gradient, hyperparams.vect, N_meta_iter, meta_alpha)
final_hyper  = hyperparams.new_vect(final_result)

#colors = ['blue', 'green', 'red', 'deepskyblue']
#index = 0
#for cur_results, name in zip(final_hyper['log_alphas'].T, parser.names):
#    if name[0] == 'weights':
#        plt.plot(np.exp(cur_results), 'o-', color = colors[index], markeredgecolor='black' )
#        print(colors[index])
#        print(name)
#        index += 1
#
#plt.show()

fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot()

colors = ['blue', 'green', 'red', 'deepskyblue']
index  = 0

ax         = fig.add_subplot()
list       = np.array(list).T
for i, (y, name) in enumerate(zip(list, parser.names) ):
    if name[0] == 'weights':
        ax.plot(np.exp(y), 'o-', color = colors[index], markeredgecolor='black')
        index +=1

ax.set_xlabel('Meta iteration')
ax.set_ylim(0,0.25)
y1 = 1.0/np.sqrt(layer_sizes[0])
y2 = 1.0/np.sqrt(layer_sizes[1])
ax.plot(ax.get_xlim(), (y2, y2), 'k--') #, label=r'$1/\sqrt{50}$')
ax.plot(ax.get_xlim(), (y1, y1), 'b--') #, label=r'$1/\sqrt{784}$')
ax.set_yticks([0.00, 1.0/np.sqrt(784), 0.10, 1.0/np.sqrt(50), 0.20, 0.25])
ax.set_yticklabels(['0.00', r"$1 / \sqrt{784}$", "0.10",
                    r"$1 / \sqrt{50}$", "0.20", "0.25"])
fig.set_size_inches((2.5,2.5))
plt.savefig('/home/marco/Documenti/Progetto_DL/initial_weights/weights_exact_rep.png')


fig.clf()
ax = fig.add_subplot(111)
index = 0
for i, (y, name) in enumerate(zip(list, parser.names) ):
    if name[0] == 'biases':
        ax.plot(np.exp(y), 'o-', color = colors[index], markeredgecolor='black')
        index +=1
ax.set_xlabel('Meta iteration')
ax.set_ylabel('Initial scale')
fig.set_size_inches((2.5,2.5))


plt.savefig('/home/marco/Documenti/Progetto_DL/initial_weights/bias_exact_rep.png')