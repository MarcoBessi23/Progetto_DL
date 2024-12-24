import autograd.numpy as np
from hyperoptimizer import data_RMD, BatchList
from neuralNet import construct_nn, WeightsParser, plot_images, make_nn_funs
from dataloader import load_data
import autograd.numpy.random as npr
from autograd import grad
import matplotlib.pyplot as plt

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
N_iters = 20
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

log_alphas  = np.full(N_iters, log_alpha_0)
gammas      = np.full(N_iters, gamma_0)


def hypergrad_fake_cp():
    pass




output = []
for i in range(N_meta_iter):
    print(f'--------------META ITERATION {i} ------------------------------')
    npr.seed(0)
    v0 = npr.randn(N_weights) * velocity_scale
    w0 = npr.randn(N_weights) * np.exp(log_param_scale)

    results = data_RMD(indexed_loss_fun, meta_loss_fun, N_iters, batch_idxs, 
                       w0, v0, gammas, np.exp(log_alphas), metas)

    results = hypergrad_fake_cp(indexed_loss_fun, meta_loss_fun, N_iters, batch_idxs, 
                       w0, v0, gammas, np.exp(log_alphas), metas)

    learning_curve = results['learning_curve']
    validation_loss = results['final_val_loss']
    fake_data_scale = np.std(hyperparser.get(metas, 'fake_data')) / true_data_scale
    test_loss = test_loss_fun(results['w_final'])
    output.append((learning_curve, validation_loss, test_loss,
                    hyperparser.get(metas, 'fake_data'), fake_data_scale,
                    np.exp(hyperparser.get(metas, 'log_L2_reg')[0])))

    metas -= results['hM_data'] * meta_stepsize
    
import matplotlib.pyplot as plt
import matplotlib


fig = plt.figure(0)
fig.clf()
ax = plt.Axes(fig, [0., 0., 1., 1.])
fig.add_axes(ax)
print('lunghezza output')
print(len(output))
all_fakedata = output[-1]
images = all_fakedata[3]
immin  = np.min(images.ravel())
immax  = np.max(images.ravel())
cax    = plot_images(images, ax, ims_per_row=5, padding=2)
cbar   = fig.colorbar(cax, ticks=[immin, 0, immax], shrink=.7)
cbar.ax.set_yticklabels(['{:2.2f}'.format(immin), '0', '{:2.2f}'.format(immax)])

plt.savefig('/home/marco/Documenti/Progetto_DL/fakeData/fake_data.png')
plt.close()
