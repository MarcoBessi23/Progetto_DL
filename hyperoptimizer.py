import autograd.numpy as np
from autograd import grad
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

def RMD_parsed(parser, hyper_vect, loss, f):

    w0, alphas, gammas = hyper_vect
    W, V   = ExactRep(w0), ExactRep(np.zeros(w0.size))
    iters  = list(zip(range(len(alphas)), alphas, gammas))
    L_grad = grad(loss)
    learning_curve = []

    for i, alpha, gamma in iters:
        print(f'forward step {i}')
        g = L_grad(W.val, i)
        cur_alpha_vect = fill_parser(parser, alpha)
        cur_gamma_vect = fill_parser(parser, gamma)
        V.mul(cur_gamma_vect)
        V.sub((1-cur_gamma_vect)*g)
        W.add(cur_alpha_vect * V.val)

    w_final = W.val
    f_grad  = grad(f)    #f corrisponde alla validation loss del paper, non nostro caso è la training loss
    d_w     = f_grad(W.val) #Questa la calcolano su tutto il training set, quindi non devo fornire indici

    d_alphas, d_gammas = np.zeros(alphas.shape), np.zeros(gammas.shape)
    d_v = np.zeros(d_w.shape)
    grad_proj = lambda w, d, i: np.dot(L_grad(w, i), d)
    L_hvp_w   = grad(grad_proj, 0)    # Returns a size(w) output.
    #L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.

    for i, alpha, gamma in iters[::-1]:
        print(f'back step{i}')
        cur_alpha_vect = fill_parser(parser, alpha)
        cur_gamma_vect = fill_parser(parser, gamma)
        for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.items()):
            d_alphas[i,j] = np.dot(d_w[ixs], V.val[ixs])

        W.sub(cur_alpha_vect * V.val)                        # Reverse position update
        g = L_grad(W.val, i)                                 # Evaluate gradient
        V.add((1.0 - cur_gamma_vect) * g)
        V.div(cur_gamma_vect)                                # Reverse momentum update
        d_v += d_w * cur_alpha_vect

        for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.items()):
            d_gammas[i,j] = np.dot(d_v[ixs], V.val[ixs] + g[ixs])

        d_w    -= L_hvp_w(W.val, (1.0 - cur_gamma_vect)*d_v, i)
        d_v    *= cur_gamma_vect


    assert np.all(ExactRep(w0).val == W.val)
    return d_w, d_alphas, d_gammas, w_final


def L2_RMD(loss, f , T, batches, w0, v0, gammas, alphas,  meta):

    W = ExactRep(w0)
    V = ExactRep(v0)

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
    learning_curve = [loss(W.val, meta, batches.all_idxs)]
    
    #forward
    for i, alpha, gamma, batch in iters:
        print(f'forward iteration {i}')
        V.mul(gamma)
        g = L_grad(W.val, meta, batch)
        V.sub((1-gamma)*g)
        W.add(alpha*V.val)        
        learning_curve.append(loss(W.val, meta, batches.all_idxs))
    
    final_params   = W.val
    dL_w = L_grad(W.val, meta, batches.all_idxs)
    dM_w = M_grad(W.val, meta)
    final_loss = loss(W.val, meta, batches.all_idxs)
    M_final    = f(final_params, meta)
    dL_v = np.zeros(dL_w.shape)
    dM_v = np.zeros(dM_w.shape)
    dL_meta = L_meta_grad(W.val, meta, batches.all_idxs)
    dM_meta = M_meta_grad(W.val, meta)
    
    for i, alpha, gamma, batch in iters[::-1]:
        print(f'backprop step {i}')

        dL_v +=  dL_w * alpha
        dM_v +=  dM_w * alpha
        
        #exact gradient descent reversion
        W.sub(alpha * V.val)
        g = L_grad(W.val, meta, batch)
        V.add((1-gamma) * g)
        V.div(gamma)
        
        dL_w -= (1-gamma)*L_hvp(W.val, dL_v, batch)
        dM_w -= (1-gamma)*L_hvp(W.val, dM_v, batch)
        dL_meta -= (1-gamma)*L_hvp_meta(W.val, meta, dL_v, batch)
        dM_meta -= (1-gamma)*L_hvp_meta(W.val, meta, dM_v, batch)
        dL_v *= gamma
        dM_v *= gamma

    assert np.all(ExactRep(w0).val == W.val)
    return {'w_final':final_params,
            'learning_curve': learning_curve,
            'final_loss':final_loss,
            'M_final': M_final,
            'param': final_params,
            'hL_w': dL_w,
            'hL_v': dL_v,
            'hM_w': dM_w,
            'hM_v': dM_v,
            'hL_meta': dL_meta,
            'hM_meta': dM_meta }





def data_RMD(loss, f , T, batches, w0, v0, gammas, alphas,  meta):

    W = ExactRep(w0)
    V = ExactRep(v0)

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
    learning_curve = [loss(W.val, meta, batches.all_idxs)]
    
    #forward
    for i, alpha, gamma, batch in iters:
        print(f'forward iteration {i}')
        V.mul(gamma)
        g = L_grad(W.val, meta, batch)
        V.sub((1-gamma)*g)
        W.add(alpha*V.val)        
        learning_curve.append(loss(W.val, meta, batches.all_idxs))
    
    final_params   = W.val
    dL_w = L_grad(W.val, meta, batches.all_idxs)
    dM_w = M_grad(W.val, meta)
    final_loss     = loss(W.val, meta, batches.all_idxs)
    final_val_loss = f(W.val, meta)
    dL_v = np.zeros(dL_w.shape)
    dM_v = np.zeros(dM_w.shape)
    dL_data = L_meta_grad(W.val, meta, batches.all_idxs)
    dM_data = M_meta_grad(W.val, meta)
    
    for i, alpha, gamma, batch in iters[::-1]:
        print(f'backprop step {i}')

        dL_v +=  dL_w * alpha
        dM_v +=  dM_w * alpha
        
        #exact gradient descent reversion
        W.sub(alpha * V.val)
        g = L_grad(W.val, meta, batch)
        V.add((1-gamma) * g)
        V.div(gamma)
        
        dL_w -= (1-gamma)*L_hvp(W.val, dL_v, batch)
        dM_w -= (1-gamma)*L_hvp(W.val, dM_v, batch)
        dL_data -= (1-gamma)*L_hvp_meta(W.val, meta, dL_v, batch)
        dM_data -= (1-gamma)*L_hvp_meta(W.val, meta, dM_v, batch)
        dL_v *= gamma
        dM_v *= gamma

    assert np.all(ExactRep(w0).val == W.val)
    return {'w_final':final_params,
            'learning_curve': learning_curve,
            'final_loss':final_loss,
            'final_val_loss':final_val_loss,
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
