import numpy as np
from hyperoptimizer import *
from neuralNet import *
from dataloader import load_data_dicts
import autograd.numpy.random as npr
from autograd import grad
from autograd.test_util import check_grads
import matplotlib.pyplot as plt
from time import time

def logit(x):
    return 1 / (1 + np.exp(-x))

def inv_logit(y):
    return -np.log( 1/y - 1)

def d_logit(x):
    return logit(x) * (1 - logit(x))

# ----- Fixed params -----
layer_sizes = [784, 50, 50, 50, 10]
batch_size = 200
N_iters = 100
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
hyperparams['log_param_scale'] = np.full(N_weight_types, init_log_param_scale)
hyperparams['log_alphas']      = np.full((N_iters, N_weight_types), init_log_alphas)
hyperparams['invlogit_gammas']  = np.full((N_iters, N_weight_types), init_invlogit_gammas)
fixed_hyperparams = VectorParser()
fixed_hyperparams['log_L2_reg'] = np.full(N_weight_types, init_log_L2_reg)
hypergrads = VectorParser()

def f_loss(w):
    return loss_fun(w, **train_data)

def hyper_gradient(hyperparams_vec, i_hyper):
    '''
    This function takes the hyperparameter vector, the meta iteration and outputs the hypergradient 
    '''
    cur_hyperparams = hyperparams.new_vect(hyperparams_vec)
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
    
    return hypergrads.vect

initial_hypergrad = hyper_gradient(hyperparams.vect, 0)
hyper = np.zeros((N_meta_iter, len(initial_hypergrad)))

for i in range(N_meta_iter):
    print(f'Meta iteration number {i}')
    hyper[i] = hyper_gradient( hyperparams.vect, i)
    
avg_hypergrad = np.mean(hyper, axis=0)
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
        ax.plot(cur_results, 'o-',  color = colors[index], markeredgecolor = 'black') #label=layer_name(name),
        index += 1
        print(np.shape(cur_results))

ax.set_ylabel('Learning rate Gradient')
ax.set_xlabel('Schedule index')
ax.set_yticks([0,])
ax.set_yticklabels(['0',])
fig.set_size_inches((6,2.5))

plt.savefig('/home/marco/Documenti/Progetto_DL/initial_hyper_values/exact_rep.png')
