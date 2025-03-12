import autograd.numpy as np
from hyperoptimizer import data_RMD, BatchList
from neuralNet import construct_nn, WeightsParser, plot_images, make_nn_funs
from dataloader import load_data_subset
import autograd.numpy.random as npr
from autograd import grad
import matplotlib.pyplot as plt
from checkpointing import BinomialCKP, adjust, maxrange, ActionType, numforw
import os

# Not going to learn:
velocity_scale = 0.0
log_alpha_0 = 0.0
gamma_0 = 0.9
log_param_scale = -4
log_L2_reg_scale = np.log(0.01)

# ----- Discrete training hyper-parameters -----
layer_sizes = [784, 10]
batch_size = 200
nSteps = 50

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

log_alphas  = np.full(nSteps, log_alpha_0)
gammas      = np.full(nSteps, gamma_0)

v0 = npr.randn(N_weights) * velocity_scale
w0 = npr.randn(N_weights) * np.exp(log_param_scale)

output = []

def hypergrad_L2_reg(loss, f , T, batches, w0, v0, gammas, alphas,  meta):
    
    w,v = np.copy(w0), np.copy(v0)
    num_epochs = T//len(batches) + 1
    iters = list(zip(range(T), alphas, gammas, batches*num_epochs))
    L_grad      = grad(loss)    # Gradient wrt parameters.
    M_grad      = grad(f)       # Gradient wrt parameters.
    L_meta_grad = grad(loss, 1) # Gradient wrt metaparameters.
    M_meta_grad = grad(f, 1)    # Gradient wrt metaparameters.
    L_hvp       = grad(lambda w, d, idxs:
                      np.dot(L_grad(w, meta, idxs), d))    # Hessian-vector product.
    L_hvp_meta  = grad(lambda w, meta, d, idxs:
                      np.dot(L_grad(w, meta, idxs), d), 1) # Returns a size(meta) output.
    
    scheduler = BinomialCKP(nSteps)
    stack     = [{} for _ in range(scheduler.snaps)]
    
    def forward(check, nfrom, nto):
        w = np.copy(stack[check]['weights'])
        v = np.copy(stack[check]['velocity'])

        for i, alpha, gamma, batch in iters[nfrom:nto]:
            print(f'forward iteration {i}')
            v *= gamma
            g  = L_grad(w, meta, batch)
            v -= (1-gamma) * g
            w += alpha * v
            
        return w, v
    
    def reverse(iteration, w, v, dL_w, dM_w, dL_v, dM_v, dL_meta, dM_meta):
        
        i, alpha, gamma, batch = iters[iteration]
        print(f'backprop step {i}')
        dL_v +=  dL_w * alpha
        dM_v +=  dM_w * alpha
        dL_w -= (1-gamma)*L_hvp(w, dL_v, batch)
        dM_w -= (1-gamma)*L_hvp(w, dM_v, batch)
        dL_meta -= (1-gamma)*L_hvp_meta(w, meta, dL_v, batch)
        dM_meta -= (1-gamma)*L_hvp_meta(w, meta, dM_v, batch)
        dL_v *= gamma
        dM_v *= gamma

        return dL_w, dM_w, dL_v, dM_v, dL_meta, dM_meta

    while True:
        action = scheduler.revolve()
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
            wF, vF = forward(scheduler.check, scheduler.oldcapo, nSteps)
            dL_w = L_grad(wF, meta, batches.all_idxs)
            dM_w = M_grad(wF, meta)
            final_loss     = loss(wF, meta, batches.all_idxs)
            final_val_loss = f(wF, meta)
            dL_v = np.zeros(dL_w.shape)
            dM_v = np.zeros(dM_w.shape)
            dL_meta = L_meta_grad(wF, meta, batches.all_idxs)
            dM_meta = M_meta_grad(wF, meta)
        elif action == ActionType.restore:
            #print(f'loading state number {scheduler.check}')
            w, v = np.copy(stack[scheduler.check]['weights']), np.copy(stack[scheduler.check]['velocity'])
            #print('valore di v che viene richiamato con restore')
            #print(v[0:4]) 
        elif action == ActionType.youturn:
            #print(f' doing reverse step at time {scheduler.fine}')
            dL_w, dM_w, dL_v, dM_v, dL_meta, dM_meta = reverse(scheduler.fine, w, v, dL_w, dM_w, dL_v, dM_v, 
                                                               dL_meta, dM_meta)
        if action == ActionType.terminate:
            break

    return {'hM_meta': dM_meta,
            'w_final': wF,
            'M_final': final_val_loss
            }

#return final_val_loss, wF, dL_w, dM_w, dL_v, dM_v, dL_meta, dM_meta



for i in range(N_meta_iter):
    print(f'---------------META ITERATION {i}----------------------------------')
    results = hypergrad_L2_reg(indexed_loss_fun, meta_loss_fun, nSteps,batch_idxs, w0, v0, gammas, 
                               np.exp(log_alphas), metas)

    validation_loss = results['M_final']
    test_loss = test_loss_fun(results['w_final'])
    output.append(( validation_loss, test_loss,
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


path_reg = os.path.join(os.getcwd(), 'regularization', 'penalties_cp.png')
plt.savefig(path_reg)

