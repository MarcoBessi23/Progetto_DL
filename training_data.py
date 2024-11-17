import numpy as np
from hyperoptimizer import *
from neuralNet import *
from dataloader import load_mnist
import autograd.numpy.random as npr
from autograd import grad
import matplotlib.pyplot as plt
from time import time

N_train_image = 10
layer_sizes = [784,10]
L2_init = 0.1
batch_size = 10
N_data = 1000
meta_iter = 50
N, train_images, train_labels, test_images, test_labels = load_mnist()
val_images = train_images[:N_data, :]
val_labels = train_labels[:N_data, :]
print(type(val_images[0]))
print(type(val_labels[0]))
train_data = np.ones((10,784))
train_labels = np.eye(10)
print(type(train_data[0]))
print(type(train_labels[0]))



batch_idxs = BatchList(N_train_image, batch_size) #10 is the number of data for the training of nn
print(batch_idxs)
alpha_0 = 0.5
gamma_0 = 0.9
N_iter = 5
alphas = np.full(N_iter, alpha_0)
gammas = np.full(N_iter, gamma_0)

data_stepsize = 0.04

parser, nn, loss = construct_nn(layer_sizes)
#parser, nn, loss = construct_nn(layer_sizes)
npr.seed(1)

w0 = npr.randn(parser.N)
v0 = npr.randn(parser.N)

def indexed_loss_fun(w, meta_params, idxs):   # To be optimized by SGD.
    return loss(w, inputs = meta_params[idxs], targets =  train_labels[idxs])
def meta_loss_fun(w):                         # To be optimized in the outer loop.
    return loss(w, inputs = val_images, targets = val_labels)

for idx in batch_idxs:
    print(np.shape(train_data[idx]))

output = []


#L_grad      = grad(indexed_loss_fun)    # Gradient wrt parameters.
#L_meta_grad = grad(loss, 1) # Gradient wrt metaparameters.
#L_hvp      = grad(lambda w, d, idxs:
#                  np.dot(L_grad(w, train_data, idxs), d))    # Hessian-vector product.
#L_hvp_meta = grad(lambda w, meta, d, idxs:
#                  np.dot(L_grad(w, meta, idxs), d), 1) 



for i in range(meta_iter):
    print(f"Meta iteration {i}")
    results = data_RMD(w = w0, v = v0, L2 = L2_init, loss = indexed_loss_fun, 
                       f = meta_loss_fun, gammas = gammas, alphas = alphas, 
                       T = 10, batches = batch_idxs, meta = train_data)
    learning_curve = results['learning_curve']
    output.append((learning_curve, train_data))
    train_data -= results['hL_data'] * data_stepsize   # Update data with one gradient step.



import matplotlib.pyplot as plt
import pickle
import matplotlib



all_learning_curves, all_fakedata = zip(*output)
fig = plt.figure(0)
fig.clf()
N_figs = 2



ax = fig.add_subplot(N_figs, 1, 1)
ax.set_title("Learning Curve")
for i, log_alphas in enumerate(all_learning_curves):
    ax.plot(log_alphas, 'o-')
ax.set_ylabel("Loss")
ax.set_xlabel("Step number")
#ax.legend(loc=4)
ax = fig.add_subplot(N_figs, 1, 2)
ax.set_title("Fake Data")
images = all_fakedata[-1]
concat_images = np.zeros((28, 0))

for i in range(N_train_image):
    cur_image = np.reshape(images[i, :], (28, 28))
    concat_images = np.concatenate((concat_images, cur_image, np.zeros((28, 5))), axis=1)


ax.matshow(concat_images, cmap = matplotlib.cm.binary)
plt.xticks(np.array([]))
plt.yticks(np.array([]))
fig.set_size_inches((8,12))
#plt.savefig("/tmp/fig.png")
#plt.savefig("fig.png")
plt.show()
