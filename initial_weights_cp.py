import autograd.numpy as np
from hyperoptimizer import data_RMD, BatchList, hyper_adam
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

layer_sizes = [784, 50, 50, 50, 10]
batch_size  = 200
nSteps   = 100
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
hyperparams['log_alphas']       = np.full((nSteps, N_weight_types), init_log_alphas)
hyperparams['invlogit_gammas']  = np.full((nSteps, N_weight_types), init_invlogit_gammas)
fixed_hyperparams = VectorParser()
fixed_hyperparams['log_L2_reg'] = np.full(N_weight_types, init_log_L2_reg)
hypergrads = VectorParser()

loss_final = []
def f_loss(w):
    return loss_fun(w, **train_data)

list_log_param_scale = []
def hyper_grad_lr(hyperparam_vec, i_hyper):

    cur_hyper = hyperparams.new_vect(hyperparam_vec)
    list_log_param_scale.append(cur_hyper['log_param_scale'].copy())
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

final_result = hyper_adam(hyper_grad_lr, hyperparams.vect, N_meta_iter, meta_alpha)
final_hyper  = hyperparams.new_vect(final_result)


fig = plt.figure(0)
fig.clf()
ax = fig.add_subplot(111)

colors = ['blue', 'green', 'red', 'deepskyblue']
index  = 0

list       = np.array(list_log_param_scale).T
for i, (y, name) in enumerate(zip(list, parser.names) ):
    if name[0] == 'weights':
        ax.plot(np.exp(y), 'o-', color = colors[index], markeredgecolor='black')
        index +=1

ax.set_xlabel('Meta iteration')
y1 = 1.0/np.sqrt(layer_sizes[0])
y2 = 1.0/np.sqrt(layer_sizes[1])
ax.plot(ax.get_xlim(), (y2, y2), 'k--') #, label=r'$1/\sqrt{50}$')
ax.plot(ax.get_xlim(), (y1, y1), 'b--') #, label=r'$1/\sqrt{784}$')
ax.set_yticks([0.00, 1.0/np.sqrt(784), 0.10, 1.0/np.sqrt(50), 0.20, 0.25])
ax.set_yticklabels(['0.00', r"$1 / \sqrt{784}$", "0.10",
                    r"$1 / \sqrt{50}$", "0.20", "0.25"])
plt.savefig('/home/marco/Documenti/Progetto_DL/initial_weights/weights_cp.png')


fig.clf()
ax = fig.add_subplot(111)
index = 0
for i, (y, name) in enumerate(zip(list, parser.names) ):
    if name[0] == 'biases':
        ax.plot(np.exp(y), 'o-', color = colors[index], markeredgecolor='black')
        index +=1
ax.set_xlabel('Meta iteration')
ax.set_ylabel('Initial scale')


plt.savefig('/home/marco/Documenti/Progetto_DL/initial_weights/bias_cp.png')