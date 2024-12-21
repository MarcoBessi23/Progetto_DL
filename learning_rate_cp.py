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
nSteps = 10
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
N_meta_iter = 5

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

    def forward(check:int, nfrom: int, nto: int):

        w = stack[check]['weights']
        v = stack[check]['velocity']

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
        '''
        proj = lambda w, d, i : np.dot(L_grad(w,i),d)
        hessianvp = grad(proj, 0)
        i, alpha, gamma = iters[iteration]

        print(f'backprop step {i}')
        cur_alpha_vect = fill_parser(parser, alpha)
        cur_gamma_vect = fill_parser(parser, gamma)
        for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.items()):
            d_alpha[i,j] = np.dot(d_w[ixs], v[ixs])
        
        #gradient descent reversion
        w -= cur_alpha_vect * v
        g  = L_grad(w, i)
        v += (1-cur_gamma_vect)*g
        v /= cur_gamma_vect

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
            wF, vF = forward(scheduler.check, scheduler.oldcapo, nSteps)
            final_loss = f(wF)
            loss_final.append(final_loss)
            #initialise gradient values
            d_alpha, d_gamma = np.zeros(alphas.shape), np.zeros(gammas.shape)
            d_w = f_grad(wF)
            d_v = np.zeros(d_w.shape)
            #first step
            d_w, d_v, d_alpha, d_gamma = reverse(nSteps-1, wF, vF, d_w, d_v, d_alpha, d_gamma)
        elif action == ActionType.restore:
            #print(f'loading state number {scheduler.check}')
            w, v = stack[scheduler.check]['weights'], stack[scheduler.check]['velocity']
        elif action == ActionType.youturn:
            #print(f' doing reverse step at time {scheduler.fine}')
            d_w, d_v, d_alpha, d_gamma = reverse(scheduler.fine, w, v, d_w, d_v, d_alpha, d_gamma)
        if action == ActionType.terminate:
            break
    
    print(w_0)
    print(stack)
    assert np.all(w_0 == w)
    weights_grad = parser.new_vect(w_0 * d_w)
    hypergrads['log_param_scale'] = [np.sum(weights_grad[name])
                                     for name in weights_grad.names]
    hypergrads['log_alphas']      =  d_alpha * alphas
    hypergrads['invlogit_gammas'] = (d_gamma * d_logit(cur_hyper['invlogit_gammas']))
    
    return hypergrads.vect           



final_result = hyper_adam(hyper_grad_lr, hyperparams.vect, N_meta_iter, meta_alpha)
final_hyper  = hyperparams.new_vect(final_result)

learning_path  = '/home/marco/Documenti/Progetto_DL/results_learning_rate/learning_schedule_cp.png'

colors = ['blue', 'green', 'red', 'deepskyblue']
index = 0
for cur_results, name in zip(final_hyper['log_alphas'].T, parser.names):
    if name[0] == 'weights':
        plt.plot(np.exp(cur_results), 'o-', color = colors[index], markeredgecolor='black' )
        print(colors[index])
        print(name)
        index += 1

plt.xlabel('Schedule index', fontdict={'family': 'serif', 'size': 12})
plt.ylabel('Learning rate', fontdict={'family': 'serif', 'size': 12})
plt.savefig(learning_path, dpi=300)
plt.close()

print(final_hyper['log_alphas'])


folder_path = '/home/marco/Documenti/Progetto_DL/results_learning_rate'
meta_learning_curve = os.path.join(folder_path, "meta_learning_cp.png")
plt.plot(loss_final, marker = 'o', color = 'blue', markeredgecolor = 'black')

plt.xlabel('meta iteration', fontdict={'family': 'serif', 'size': 12})
plt.ylabel('loss', fontdict={'family': 'serif', 'size': 12})

plt.savefig(meta_learning_curve, dpi=300)
plt.close()





#def adam_single_iter(hypergrad, hyper, iter, im, iv, step_size=0.1,
#                      b1 = 0.1, b2 = 0.01, eps = 1e-4, lam=1e-4):
#    '''
#    im, iv sono stati inizializzati a 0, a questa funzione vengono
#    passati gli iper di alpha e di gamma e aggiorna i due iper separatamente
#    '''
#
#    b1t    = 1 - (1-b1)*(lam**iter)
#    im     = b1t*hypergrad     + (1-b1t)*im   # First  moment estimate
#    iv     = b2*(hypergrad**2) + (1-b2)*iv    # Second moment estimate
#    mhat   = im/(1-(1-b1)**(iter+1))          # Bias correction
#    vhat   = iv/(1-(1-b2)**(iter+1))
#    hyper -= step_size*mhat/(np.sqrt(vhat) + eps)
#
#    return hyper, im, iv

#def adam_single_iter(hypergrad, hyper, iter, im, iv, step_size=0.1, b1 = 0.1, b2 = 0.01, eps = 1e-4, lam=1e-4):
#    '''
#    im, iv sono stati inizializzati a 0, a questa funzione vengono
#    passati gli iper di alpha e di gamma e aggiorna i due iper separatamente
#    '''
#    im = (1 - b1) * hypergrad      + b1 * im  # First  moment estimate.
#    iv = (1 - b2) * (hypergrad**2) + b2 * iv  # Second moment estimate.
#    mhat = im / (1 - b1 ** (iter + 1))  # Bias correction.
#    vhat = iv / (1 - b2 ** (iter + 1))
#    hyper -= step_size*mhat/(np.sqrt(vhat) + eps)
#
#    return hyper, im, iv
#
##################################################################################################################
#################         OSSERVAZIONE:                                                ###########################
#################         CALCOLANDO LA TRAINING LOSS SU TUTTO IL TRAINING SET         ###########################
#################         RIMANGONO FUORI CIRCA 1300 INDICI OGNI META ITER             ###########################
##################################################################################################################
####insieme = set()
####my_set = set()
####my_set.update(all_idxs)
####i_hyper = 4  #provare anche con altri indici
####for i in range(100):
####    rs = RandomState((seed, i_hyper, i))
####    batch = rs.randint(N_train, size=batch_size)
####    insieme.update(batch)
####
####
####print('confronto insiemi')
####print(len(my_set)-len(insieme))


#imp, ivp   = np.zeros_like(hyperparams['log_param_scale']), np.zeros_like(hyperparams['log_param_scale'])
#ima, iva   = np.zeros_like(hyperparams['log_alphas']     ), np.zeros_like(hyperparams['log_alphas']     )
#img, ivg   = np.zeros_like(hyperparams['invlogit_gammas']), np.zeros_like(hyperparams['invlogit_gammas'])
#loss_final = []
#for i in range(meta_iter):
#    print(f'------------------------META ITERATION {i}--------------------------------------')
#
#    res = hyper_grad_lr(nSteps, parser, indexed_loss_fun, indexed_loss_fun, hyperparams.vect, i)
#    if res == 'NAN':
#        print('NAN da qualche parte')
#        break
#    else:
#        hypergrad_alpha  = np.exp(hyperparams['log_alphas']) * res[2]      #derivate rispetto ad hyper
#        hypergrad_gamma  = d_logit(hyperparams['invlogit_gammas'])* res[3] #derivate rispetto ad hyper
#        grad_param_scale = parser.new_vect(res[0])
#        l = [np.sum(grad_param_scale[name]) for name in parser.names]
#        hypergrad_param_scale = np.array(l)
#        loss_final.append(res[4])
#
#    #lavori nello spazio logaritmico
#    hyperparams['invlogit_gammas'], ima, iva = adam_single_iter(hypergrad_gamma, hyperparams['invlogit_gammas'],
#                                                                 i, img, ivg, step_size= meta_step_size)
#    hyperparams['log_alphas'],      img, ivg = adam_single_iter(hypergrad_alpha, hyperparams['log_alphas'], 
#                                                                i, ima, iva, step_size= meta_step_size)
#    hyperparams['log_param_scale'], imp, ivp = adam_single_iter(hypergrad_param_scale, hyperparams['log_param_scale'], 
#                                                                i, imp, ivp, step_size= meta_step_size)
#
#colors = ['blue', 'green', 'red', 'deepskyblue']
#index = 0
#for hyp, name in  zip(hyperparams['log_alphas'].T, parser.names):
#    if name[0] == 'weights':
#        plt.plot(np.exp(hyp), marker = 'o', label = name[0], color = colors[index], markeredgecolor='black')
#        plt.xlabel('Schedule index', fontdict={'family': 'serif', 'size': 12})
#        plt.ylabel('Learnin rate', fontdict={'family': 'serif', 'size': 12})
#        print(colors[index])
#        print(name)
#        print(np.exp(hyp))
#        index +=1







#vect = [np.float64(0.5098528822537987), np.float64(0.45007951789311496), np.float64(0.40614206553898463),
#        np.float64(0.320914297000315), np.float64(0.3210824506699599), np.float64(0.2918604952080652), 
#        np.float64(0.2889029233544971), np.float64(0.29662510410128384), np.float64(0.2697185632195829), 
#        np.float64(0.26096429785565173), np.float64(0.24581436246278301), np.float64(0.24901208939515532), 
#        np.float64(0.23833014579167033), np.float64(0.23800255023006178), np.float64(0.22813667966569884), 
#        np.float64(0.22422204687244887), np.float64(0.21905510317599083), np.float64(0.20839324576234064), 
#        np.float64(0.20107722015666069), np.float64(0.20034656315304547), np.float64(0.19714751628759125), 
#        np.float64(0.18226048809293685), np.float64(0.17817883482473182), np.float64(0.18200727682843873), 
#        np.float64(0.1727017088493723), np.float64(0.16819194500260715), np.float64(0.159805015362057), 
#        np.float64(0.14960428409565885), np.float64(0.14195492520244588), np.float64(0.14966959874744035), 
#        np.float64(0.1464749777289587), np.float64(0.13973227239583352), np.float64(0.13987845883621344), 
#        np.float64(0.12039719132129532), np.float64(0.12847400966401978), np.float64(0.1289775610590841), 
#        np.float64(0.13206958912736821), np.float64(0.11965431368637666), np.float64(0.13256559275865618), 
#        np.float64(0.12127299241835535), np.float64(0.11652436230113533), np.float64(0.13006518200003064), 
#        np.float64(0.11152387501012526), np.float64(0.10629773362453449), np.float64(0.1330690653046463), 
#        np.float64(0.11477889452771979), np.float64(0.11807823592037583), np.float64(0.11314421261875371),
#        np.float64(0.12441242224932747), np.float64(0.1481984019395128)]
#
#import matplotlib.ticker as ticker
#loss_vector = np.array(vect) 
#x = np.arange(len(loss_vector))
#plt.plot(x, loss_vector, marker = 'o', color = 'blue', markeredgecolor = 'black')
#plt.xlabel('meta iteration', fontdict={'family': 'serif', 'size': 12})
#plt.ylabel('loss', fontdict={'family': 'serif', 'size': 12})
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{x:.1f}'))
#plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
#_,high = plt.gca().get_ylim()
#plt.gca().set_ylim([0, high])
#plt.show()