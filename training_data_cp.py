import autograd.numpy as np
from hyperoptimizer import data_RMD, BatchList
from neuralNet import construct_nn, WeightsParser, plot_images, make_nn_funs
from dataloader import load_data
import autograd.numpy.random as npr
from autograd import grad
import matplotlib.pyplot as plt
from checkpointing import BinomialCKP, adjust, maxrange, ActionType, numforw
import os


# ----- Initial values of continuous hyper-parameters -----
init_log_L2_reg = np.log(0.01)

one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)


# Not going to learn:
velocity_scale = 0.0
log_alpha_0 = 0.0
gamma_0 = 0.9
log_param_scale = -4

# ----- Discrete training hyper-parameters -----
layer_sizes = [784, 10]
batch_size = 10
nSteps = 20
N_classes = 10

# ----- Variables for meta-optimization -----
N_fake_data = 10
fake_data_L2_reg = 0.0
N_val_data = 10000
N_test_data = 1000
meta_stepsize = 1
N_meta_iter = 40
init_fake_data_scale = 0.01



val_images, val_labels, test_images, test_labels, _ = load_data(normalize=True)
val_images = val_images[:N_val_data, :]
val_labels = val_labels[:N_val_data, :]
true_data_scale = np.std(val_images)

test_images = test_images[:N_test_data, :]
test_labels = test_labels[:N_test_data, :]
batch_idxs  = BatchList(N_fake_data, batch_size)
parser, _, loss_fun = construct_nn(layer_sizes)
N_weights   = parser.N

npr.seed(0)

init_fake_data = npr.randn(*(val_images[:N_fake_data, :].shape)) * init_fake_data_scale
print('fake data shape')
print(np.shape(init_fake_data))
fake_labels = one_hot(np.array(range(N_fake_data)) % N_classes, N_classes)  # One of each.

hyperparser = WeightsParser()
hyperparser.add_weights('log_L2_reg', (1,))
hyperparser.add_weights('fake_data', init_fake_data.shape)
metas = np.zeros(hyperparser.N)
hyperparser.set(metas, 'log_L2_reg', init_log_L2_reg)
hyperparser.set(metas, 'fake_data', init_fake_data)

def indexed_loss_fun(w, meta_params, idxs):   # To be optimized by SGD.
    L2_reg    = np.exp(hyperparser.get(meta_params, 'log_L2_reg')[0])
    fake_data = hyperparser.get(meta_params, 'fake_data')
    return loss_fun(w, X = fake_data[idxs], T = fake_labels[idxs], L2_reg = L2_reg)

def meta_loss_fun(w, meta_params):            # To be optimized in the outer loop.
    fake_data = hyperparser.get(meta_params, 'fake_data')
    log_prior = -fake_data_L2_reg * np.dot(fake_data.ravel(), fake_data.ravel())
    return loss_fun(w, X = val_images, T = val_labels) - log_prior

def test_loss_fun(w):                         # To measure actual performance.
    return loss_fun(w, X = test_images, T = test_labels)

log_alphas  = np.full(nSteps, log_alpha_0)
gammas      = np.full(nSteps, gamma_0)


def hypergrad_fake_cp(loss, f , T, batches, w0, v0, gammas, alphas,  meta):
    
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
    #learning_curve = [loss(w, meta, batches.all_idxs)]



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

    def reverse(iteration, w, v, dL_w, dM_w, dL_v, dM_v, dL_data, dM_data):
        
        i, alpha, gamma, batch = iters[iteration]
        print(f'backprop step {i}')
        dL_v +=  dL_w * alpha
        dM_v +=  dM_w * alpha
        dL_w -= (1-gamma)*L_hvp(w, dL_v, batch)
        dM_w -= (1-gamma)*L_hvp(w, dM_v, batch)
        dL_data -= (1-gamma)*L_hvp_meta(w, meta, dL_v, batch)
        dM_data -= (1-gamma)*L_hvp_meta(w, meta, dM_v, batch)
        dL_v *= gamma
        dM_v *= gamma

        return dL_w, dM_w, dL_v, dM_v, dL_data, dM_data
    
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
            dL_data = L_meta_grad(wF, meta, batches.all_idxs)
            dM_data = M_meta_grad(wF, meta)
        elif action == ActionType.restore:
            #print(f'loading state number {scheduler.check}')
            w, v = np.copy(stack[scheduler.check]['weights']), np.copy(stack[scheduler.check]['velocity'])
            #print('valore di v che viene richiamato con restore')
            #print(v[0:4]) 
        elif action == ActionType.youturn:
            #print(f' doing reverse step at time {scheduler.fine}')
            dL_w, dM_w, dL_v, dM_v, dL_data, dM_data = reverse(scheduler.fine, w, v, dL_w, dM_w, dL_v, dM_v, 
                                                               dL_data, dM_data)
        if action == ActionType.terminate:
            break

    
    return final_val_loss, wF, dL_w, dM_w, dL_v, dM_v, dL_data, dM_data




output = []
for i in range(N_meta_iter):
    print(f'--------------META ITERATION {i} ------------------------------')
    npr.seed(0)
    v0 = npr.randn(N_weights) * velocity_scale
    w0 = npr.randn(N_weights) * np.exp(log_param_scale)

    results = hypergrad_fake_cp(indexed_loss_fun, meta_loss_fun, nSteps, batch_idxs, 
                       w0, v0, gammas, np.exp(log_alphas), metas)

    validation_loss = results[0]
    fake_data_scale = np.std(hyperparser.get(metas, 'fake_data')) / true_data_scale
    test_loss = test_loss_fun(results[1])
    output.append((validation_loss, test_loss,
                    hyperparser.get(metas, 'fake_data'), fake_data_scale,
                    np.exp(hyperparser.get(metas, 'log_L2_reg')[0])))

    metas -= results[7] * meta_stepsize
    
import matplotlib.pyplot as plt
import matplotlib


fig = plt.figure(0)
fig.clf()
ax = plt.Axes(fig, [0., 0., 1., 1.])
fig.add_axes(ax)
print('lunghezza output')
print(len(output))
all_fakedata = output[-1]
images = all_fakedata[2]
immin  = np.min(images.ravel())
immax  = np.max(images.ravel())
cax    = plot_images(images, ax, ims_per_row=5, padding=2)
cbar   = fig.colorbar(cax, ticks=[immin, 0, immax], shrink=.7)
cbar.ax.set_yticklabels(['{:2.2f}'.format(immin), '0', '{:2.2f}'.format(immax)])

path_data = os.path.join(os.getcwd(), 'fakeData', 'fake_data_cp.png')
plt.savefig(path_data)
plt.close()
