import numpy as np
from autograd import grad
from fractions import Fraction


RADIX_SCALE = 2**52

class BitStore(object):
    """Efficiently stores information with non-integer number of bits (up to 16)."""
    def __init__(self, length:int):
        # Use an array of Python 'long' ints which conveniently grow
        # as large as necessary. It's about 50X slower though...
        self.store = np.array([0] * length, dtype=object)

    def push(self, N:int, M:int):
        """Stores integer N, given that 0 <= N < M"""
        assert M <= 2**16
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
    #    assert a > 0.0
    #    d = 2**16 // int(a + 1)  #// instead of / because d must have type int
    #    n = int(a * d + 1)
    #    print(n)
    #    print(d)
    #    return  n, d


    def float_to_rational(self, a):
        assert a > 0.0
        frac = Fraction(a).limit_denominator(65536)
        return frac.numerator, frac.denominator

    def float_to_intrep(self, x):
        return (x * RADIX_SCALE).astype(np.int64)

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
    
    for _, alpha, gamma, batch in iters:
        g = gradient(W.val, batch)
        V.mul(gamma)
        V.sub((1-gamma)*g)
        W.add(alpha*V.val)
    
    print('calcolo dw')
    l_grad = grad(f)
    d_w = l_grad(W.val,batches.all_idxs)
    print('calcolo hyper grad')
    hyper_gradient =grad(lambda w, idx, d: np.dot(gradient(w,idx),d))
    print(np.shape(d_w))

    d_alpha = []
    d_gamma = []
    d_v = np.zeros_like(w)

    
    #backprop
    for t, alpha, gamma, batch in iters[::-1]:
        print('inizio a calcolare la backprop degli iperparametri')
        print(f'backprop passo {t}')
        d_alpha.append(np.dot(d_w,V.val))

        v = V.val
        w = W.val
        #exact gradient descent reversion
        g = gradient(W.val, batch)
        W.sub(alpha*V.val)
        V.add((1-gamma)*g)
        V.div(gamma)

        print(np.shape(d_w))
        
        print('aggiorno dv')
        d_v += alpha*d_w
        print('aggiungo a gamma')
        d_gamma.append(np.dot(d_v,v+g))
        print('aggiorno dw')
        d_w -= (1-gamma)*hyper_gradient(w, batch, d_v)
        d_v *= gamma
    
    d_alpha = np.array(d_alpha)
    d_gamma = np.array(d_gamma)

    return d_w, d_v, d_gamma, d_alpha

#np.random.seed(42)
#v = np.random.rand(5)
#print(v)
#gamma = 2.123
#V = ExactRep(v)
#print(V)
#
#V.mul(gamma)
#print('valore di V dopo moltiplicazione:')
#print(V.val)
#print('valore di v per gamma:')
#print(v*gamma)
#print('valore di v per gamma approssimato a razionale:')
#g = 46377/21845
#print(v*g)