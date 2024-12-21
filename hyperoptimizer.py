import autograd.numpy as np
from autograd import grad
from fractions import Fraction
from collections import deque
from time import time
from neuralNet import VectorParser, fill_parser
from ExactRep import BitStore, ExactRep

class BatchList(list):
    def __init__(self, N_total, N_batch):
        start = 0
        while start < N_total:
            self.append(slice(start, start + N_batch))
            start += N_batch
        self.all_idxs = slice(0, N_total)

def RMD(w, v, loss, f, gammas, alphas, T, batches):

    '''
    SGD with momentum and reverse mode differentiation
    input:
    w: model parameter
    v: velocity parameter
    loss: training loss
    f: validation loss
    alphas: step size hypers
    gammas: velocity update hyper
    theta: regularization hyper
    T: total number of iterations
    output:
    The gradient of validation function wrt the initial w and v, gammas and alphas
    '''

    W = ExactRep(w)
    V = ExactRep(v)
    num_epochs = int(T/len(batches)) + 1
    gradient = grad(loss)
    iters = list(zip(range(T), alphas, gammas, batches * num_epochs))
    learning_curve = []

    #forward
    for i, alpha, gamma, batch in iters:
        print(f'forward iteration {i}')
        g = gradient(W.val, batch)
        V.mul(gamma)
        V.sub((1-gamma)*g)
        W.add(alpha*V.val)
        learning_curve.append(loss(W.val, batches.all_idxs))

    final_loss = loss(W.val, batches.all_idxs)
    final_param = W.val

    l_grad = grad(f)
    d_w = l_grad(W.val,batches.all_idxs)
    hyper_gradient = grad(lambda w, idx, d: np.dot(gradient(w,idx),d))

    d_alpha = deque()
    d_gamma = deque()
    d_v = np.zeros_like(w)

    #backprop 
    for t, alpha, gamma, batch in iters[::-1]:
        print(f'backprop step {t}')
        d_alpha.appendleft(np.dot(d_w,V.val))

        #exact gradient descent reversion
        g = gradient(W.val, batch)
        W.sub(alpha*V.val)
        V.add((1-gamma)*g)
        V.div(gamma)

        d_v += alpha*d_w
        d_gamma.appendleft(np.dot(d_v,V.val+g))
        d_w -= (1-gamma)*hyper_gradient(W.val, batch, d_v)
        d_v *= gamma

    d_alpha = np.array(d_alpha)
    d_gamma = np.array(d_gamma)

    return {'learning curve': learning_curve,
            'loss':final_loss,
            'param': final_param,
            'hg_w':d_w,
            'hg_v': d_v,
            'hg_gamma':d_gamma,
            'hg_alpha':d_alpha   }

def load_alpha(parser, alpha):
    index_shape_vec = parser.shape_idx
    cur_alpha = np.zeros(parser.N)
    for i, (val, _) in enumerate(index_shape_vec.values()):
        cur_alpha[val] = alpha[i]
    return cur_alpha

def RMD_parsed(parser, hyper_vect, loss, indices):
    
    w0, alphas, gammas = hyper_vect
    W, V = ExactRep(w0), ExactRep(np.zeros(w0.size))
    iters = list(zip(range(len(alphas)), alphas, gammas))
    L_grad = grad(loss)
    learning_curve = []
    for i, alpha, gamma in iters:
        g = L_grad(W.val, i)
        cur_alpha_vect = fill_parser(parser, alpha)
        cur_gamma_vect = fill_parser(parser, gamma)
        V.mul(cur_gamma_vect)
        V.sub((1-cur_gamma_vect)*g)
        W.add(cur_alpha_vect * V.val)

    w_final = W.val
    d_w = L_grad(W.val, indices) #Bisogna trovare il modo di passare tutti gli indici su cui è stata addestrata

    d_alphas, d_gammas = np.zeros(alphas.shape), np.zeros(gammas.shape)
    d_v = np.zeros(d_w.shape)
    grad_proj = lambda w, d, i: np.dot(L_grad(w, i), d)
    L_hvp_w   = grad(grad_proj, 0)  # Returns a size(x) output.
    #L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.

    for i, alpha, gamma in iters[::-1]:
        print(f'back step{i}')
        cur_alpha_vect = fill_parser(parser, alpha)
        cur_gamma_vect  = fill_parser(parser, gamma)
        for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.items()):
            d_alphas[i,j] = np.dot(d_w[ixs], V.val[ixs])

        W.sub(cur_alpha_vect * V.val)                        # Reverse position update
        g = L_grad(W.val, i)                                 # Evaluate gradient
        V.add((1.0 - cur_gamma_vect) * g)
        V.div(cur_gamma_vect)  # Reverse momentum update

        d_v += d_w * cur_alpha_vect

        for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.items()):
            d_gammas[i,j] = np.dot(d_v[ixs], V.val[ixs] + g[ixs])

        d_w    -= L_hvp_w(W.val, (1.0 - cur_gamma_vect)*d_v, i)
        d_v    *= cur_gamma_vect
    
    
    assert np.all(ExactRep(w0).val == W.val)
    return d_w, d_alphas, d_gammas, w_final

def multi_RMD(w, v, parser, loss, f, gammas, alphas, T, batches, only_forw):
    '''
    Version of RMD in which learning rate has shape like neural network parameters
    '''
    W = ExactRep(w)
    V = ExactRep(v)
    num_epochs = int(T/len(batches)) + 1
    gradient = grad(loss)
    iters = list(zip(range(T), alphas, gammas, batches * num_epochs))
    learning_curve = []

    #forward
    for i, alpha, gamma, batch in iters:
        print(f'forward iteration {i}')
        g = gradient(W.val, batch)
        cur_alpha = load_alpha(parser, alpha)
        cur_gamma = load_alpha(parser, gamma)
        V.mul(cur_gamma)
        V.sub((1-cur_gamma)*g)
        W.add(cur_alpha * V.val)
        learning_curve.append(loss(W.val, batches.all_idxs))

    final_loss = loss(W.val, batches.all_idxs)
    final_param = W.val

    if only_forw:
        return final_loss
    
    l_grad = grad(f)

    d_w = l_grad(W.val,batches.all_idxs)
    hessianvp = grad(lambda w, idx, d: np.dot(gradient(w,idx),d))

    d_alpha, d_gamma = np.zeros(alphas.shape), np.zeros(gammas.shape)
    d_v = np.zeros_like(w)

    #backprop
    for i, alpha, gamma, batch in iters[::-1]:
        print(f'backprop step {i}')
        cur_alpha = load_alpha(parser, alpha)
        cur_gamma = load_alpha(parser, gamma)
        for j, (ixs, _) in enumerate(parser.shape_idx.values()):
                d_alpha[i,j] = np.dot(d_w[ixs], V.val[ixs])

        #exact gradient descent reversion
        g = gradient(W.val, batch)
        W.sub(cur_alpha * V.val)
        V.add((1-cur_gamma)*g)
        V.div(cur_gamma)

        d_v += cur_alpha*d_w
        for j, (ixs, _) in enumerate(parser.shape_idx.values()):
                d_gamma[i,j] = np.dot(d_v[ixs], V.val[ixs] + g[ixs])
        #d_gamma[i] = np.dot(d_v,V.val+g)
        d_w -= (1-cur_gamma)*hessianvp(W.val, batch, d_v)
        d_v *= cur_gamma

        print('end backprop')

    return {'learning curve': learning_curve,
            'loss':final_loss,
            'param': final_param,
            'hg_w':d_w,
            'hg_v': d_v,
            'hg_gamma':d_gamma,
            'hg_alpha':d_alpha   }


def L2_RMD(w, v, L2, loss, f, gammas, alphas, T, batches, only_forward = True):

    '''
    SGD with momentum and reverse mode differentiation
    input:
    w: model parameter
    v: velocity parameter
    loss: training loss
    f: validation loss
    alphas: step size hypers
    gammas: velocity update hyper
    theta: regularization hyper
    T: total number of iterations
    output:
    The gradient of validation function wrt the initial w and v, gammas and alphas
    '''

    W = ExactRep(w)
    V = ExactRep(v)
    num_epochs = int(T/len(batches)) + 1
    gradient = grad(loss)
    iters = list(zip(range(T), alphas, gammas, batches * num_epochs))
    learning_curve = []

    #forward
    for i, alpha, gamma, batch in iters:
        print(f'forward iteration {i}')
        g = gradient(W.val, L2, batch)
        print(type(g))
        #print('gradiente')
        #print(np.max(g))
        #print(np.isnan(g).any())
        V.mul(gamma)
        #print('V1')
        #print(np.isnan(V.val).any())
        V.sub((1-gamma)*g)
        #print('V2')
        #print(np.isnan(V.val).any())
        W.add(alpha*V.val)
        #print('W')
        #print(np.max(W.val))
        learning_curve.append(loss(W.val, L2, batches.all_idxs))

    final_loss = loss(W.val, L2, batches.all_idxs)
    final_param = W.val

    if only_forward:
        return final_param

    l_grad = grad(f)
    d_w = l_grad(W.val, L2, batches.all_idxs)
    fun = lambda w, L2, idx, d: np.dot(l_grad(w, L2, idx),d)
    hyper_gradient_w = grad(fun, 0)
    hyper_gradient_L2 = grad(fun, 1)

    d_v = np.zeros_like(w)
    d_L2 = np.zeros_like(w)


    #backprop 
    for t, alpha, gamma, batch in iters[::-1]:
        print(f'backprop step {t}')
    
        #exact gradient descent reversion
        g = gradient(W.val, L2, batch)
        print('Maximum Gradient')
        print(np.max(g))
        print('w')
        W.sub(alpha*V.val)
        print('V1')
        V.add((1-gamma)*g)
        print('V2')
        V.div(gamma)
        print(np.max(V.val))
        

        print('d_v')
        d_v += alpha*d_w
        print(np.max(d_v))
        print('d_w')
        d_w -= (1-gamma)*hyper_gradient_w(W.val, L2, batch, d_v)
        print(np.max(d_w))
        print('d_L2')
        d_L2 -= (1-gamma)*hyper_gradient_L2(W.val, L2, batch, d_v)
        print(np.max(d_L2))
        print('gamma x d_v')
        d_v *= gamma
        print(d_v)



    return {'learning curve': learning_curve,
            'loss':final_loss,
            'param': final_param,
            'hg_w':d_w,
            'hg_v': d_v,
            'hg_L2': d_L2}


def data_RMD(w, v, L2, loss, f , gammas, alphas, T, batches, meta):

    W = ExactRep(w)
    V = ExactRep(v)

    iter_per_epoch = len(batches)
    num_epochs = int(T/len(batches)) + 1
    iters = list(zip(range(T), alphas, gammas, batches*num_epochs))
    
    learning_curve = []
    validation_curve = []
    L_grad      = grad(loss)    # Gradient wrt parameters.
    M_grad      = grad(f)       # Gradient wrt parameters.
    L_meta_grad = grad(loss, 1) # Gradient wrt metaparameters.
    M_meta_grad = grad(f, 1)    # Gradient wrt metaparameters.
    L_hvp      = grad(lambda w, d, idxs:
                      np.dot(L_grad(w, meta, idxs), d))    # Hessian-vector product.
    L_hvp_meta = grad(lambda w, meta, d, idxs:
                      np.dot(L_grad(w, meta, idxs), d), 1) # Returns a size(meta) output.
    learning_curve = [loss(W.val, meta, batches.all_idxs)]
    #forward
    for i, alpha, gamma, batch in iters:
        print(f'forward iteration {i}')
        g = L_grad(W.val, meta, batch)
        V.mul(gamma)
        V.sub((1-gamma)*g)
        W.add(alpha*V.val)        
        learning_curve.append(loss(W.val, meta, batches.all_idxs))
        validation_curve.append(loss(W.val, meta, batches.all_idxs))

    final_loss = loss(W.val, meta, batches.all_idxs)
    final_params = W.val
    dL_w = L_grad(W.val, meta, batches.all_idxs)
    dL_v = np.zeros(dL_w.shape)
    dM_w = M_grad(W.val)            #maybe you have to change meta_loss to take meta as a variable
    dM_v = np.zeros(dL_w.shape)
    dL_data = L_meta_grad(W.val, meta, batches.all_idxs)
    dM_data = np.zeros(dL_data.shape)
    #dM_data = M_meta_grad(W.val)    #same as above

    for i, alpha, gamma, batch in iters[::-1]:
        print(f'backprop step {i}')
    
        #exact gradient descent reversion
        g = L_grad(W.val, meta, batch)
        W.sub(alpha*V.val)
        V.add((1-gamma) * g)
        V.div(gamma)
        dL_v += alpha * dL_w
        dM_v += alpha * dM_w
        dL_w -= (1-gamma)*L_hvp(W.val, dL_v, batch)
        dM_w -= (1-gamma)*L_hvp(W.val, dM_v, batch)
        dL_data -= (1-gamma)*L_hvp_meta(W.val, meta, dL_v, batch)
        dM_data = (1-gamma)*L_hvp_meta(W.val, meta, dM_v, batch)
        #dM_data -= (1-gamma)*L_hvp_meta(W.val, L2, dM_v, batch)
        dL_v *= gamma
        dM_v *= gamma
    
    return {'learning_curve': learning_curve,
            'validation_curve': validation_curve,
            'loss':final_loss,
            'param': final_params,
            'hL_w': dL_w,
            'hL_v': dL_v,
            'hM_w': dM_w,
            'hM_v': dM_v,
            'hL_data': dL_data,
            'hM_data': dM_data }



def hyper_adam(grad, x, num_iters=100,
         step_size = 0.1, b1 = 0.1, b2 = 0.01, eps = 10**-4, lam=10**-4):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        print(f'::::::::::::::META ITERATION {i}::::::::::::::::::::')
        b1t = 1 - (1-b1)*(lam**i)
        g = grad(x, i)

        m = b1t*g     + (1-b1t)*m   # First  moment estimate
        v = b2*(g**2) + (1-b2)*v    # Second moment estimate
        mhat = m/(1-(1-b1)**(i+1))  # Bias correction
        vhat = v/(1-(1-b2)**(i+1))
        x -= step_size*mhat/(np.sqrt(vhat) + eps)
    return x
