import numpy as np
from autograd import grad
from fractions import Fraction
from collections import deque
from time import time

RADIX_SCALE = 2**52

class BitStore(object):
    """Efficiently stores information with non-integer number of bits (up to 16)."""
    def __init__(self, length:int):
        self.store = np.array([0] * length, dtype=object)

    def push(self, N:int, M:int):
        """Stores integer N, given that 0 <= N < M"""
        assert np.all(M <= 2**16)
        self.store *= M
        self.store += N

    def pop(self, M):
        """Retrieves the last integer stored."""
        N = self.store % M
        self.store /= M
        return N


class ExactRep(object):
    """Fixed-point representation of arrays with auxilliary bits such
    that + - * / ops are all exactly invertible (except for
    overflow)."""
    def __init__(self, val, from_intrep=False):
        if from_intrep:
            self.intrep = val
        else:
            self.intrep = self.float_to_intrep(val)

        self.aux = BitStore(len(val))

    def add(self, A):
        """Reversible addition of vector or scalar A."""
        self.intrep += self.float_to_intrep(A)

    def sub(self, A):
        self.add(-A)

    def rational_mul(self, n:int, d:int):
        self.aux.push(self.intrep % d, d) # Store remainder bits externally
        self.intrep //= d                 # Divide by denominator
        self.intrep *= n                  # Multiply by numerator
        self.intrep += np.array(self.aux.pop(n), dtype=np.int64)
        #self.intrep += self.aux.pop(n)    # Pack bits into the remainder

    def mul(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(n, d)
        
    def div(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(d, n)

    #def float_to_rational(self, a):
    #    d = 2**16 // int(a + 1)  #// instead of / because d must have type int
    #    n = int(a * d + 1)
    #    return  n, d

    def float_to_rational(self, a):
        assert np.all(a > 0.0)
        d = 2**16 // np.fix(a+1).astype(int) # Uglier than it used to be: np.int(a + 1)
        n = np.fix(a * d + 1).astype(int)
        return  n, d

    #def float_to_rational(self, a):
    #    assert np.all(a > 0.0)
    #    frac = Fraction(a).limit_denominator(65536)
    #    return frac.numerator, frac.denominator

    def float_to_intrep(self, x):
        val = x*RADIX_SCALE
        if np.isnan(val).any() or np.isinf(val).any():
            print("Valori non validi trovati in x_scaled")
            print("Indice dei valori NaN:", np.where(np.isnan(val)))
            print("Indice dei valori inf:", np.where(np.isinf(val)))
            print("Valori problematici:", x[np.isnan(val) | np.isinf(val)])
            raise ValueError("Valori non validi in x_scaled")
        
        return val.astype(np.int64)   #(x * RADIX_SCALE).astype(np.int64)

    @property
    def val(self):
        return self.intrep.astype(np.float64) / RADIX_SCALE

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
    for i, (val1, _) in enumerate(index_shape_vec.values()):
        cur_alpha[val1] = alpha[i]
    return cur_alpha


def multi_RMD(w, v, parser, loss, f, gammas, alphas, T, batches):
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

    l_grad = grad(f)

    d_w = l_grad(W.val,batches.all_idxs)
    hyper_gradient = grad(lambda w, idx, d: np.dot(gradient(w,idx),d))

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
        d_w -= (1-cur_gamma)*hyper_gradient(W.val, batch, d_v)
        d_v *= cur_gamma

        print('end backprop')

    return {'learning curve': learning_curve,
            'loss':final_loss,
            'param': final_param,
            'hg_w':d_w,
            'hg_v': d_v,
            'hg_gamma':d_gamma,
            'hg_alpha':d_alpha   }


def L2_RMD(w, v, L2, loss, f, gammas, alphas, T, batches):

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
    d_w = l_grad(W.val, batches.all_idxs)
    hyper_gradient = grad(lambda w, idx, d: np.dot(gradient(w,idx),d))
    
    
    d_v = np.zeros_like(w)
    d_L2 = np.zeros_like(w)


    #backprop 
    for t, alpha, gamma, batch in iters[::-1]:
        print(f'backprop step {t}')
    
        #exact gradient descent reversion
        g = gradient(W.val, batch)
        W.sub(alpha*V.val)
        V.add((1-gamma)*g)
        V.div(gamma)

        d_v += alpha*d_w
        d_w -= (1-gamma)*hyper_gradient(W.val, batch, d_v)
        d_L2 -= (1-gamma)*hyper_gradient(W.val, batch, d_v)
        d_v *= gamma



    return {'learning curve': learning_curve,
            'loss':final_loss,
            'param': final_param,
            'hg_w':d_w,
            'hg_v': d_v,
            }
