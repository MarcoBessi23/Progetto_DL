"""A multi-layer perceptron for classification of MNIST handwritten digits."""
#import numpy as np
import autograd.numpy as np 
#import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam, sgd
#from autograd.scipy.special import logsumexp
import numpy.random as npr
import hashlib
from collections import OrderedDict


class RandomState(npr.RandomState):
    """Takes an arbitrary object as seed (uses its string representation)"""
    def __init__(self, obj):
        hashed_int = int(hashlib.md5(str(obj).encode()).hexdigest()[:8], base=16)
        super(RandomState, self).__init__(hashed_int)


def logsumexp(X, axis):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))

class Weight_Parser():
    def __init__(self) :
        self.shape_idx = {}
        self.N = 0
    
    def add(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.shape_idx[name] = (slice(start, self.N),shape) #salva gli indici dei pesi che vanno nel layer e la forma

    def get(self, name, param):
        idxs, shape = self.shape_idx[name]
        return np.reshape(param[idxs], shape)

class VectorParser(object):
    def __init__(self):
        self.idxs_and_shapes = OrderedDict()
        self.vect = np.zeros((0,))

    def add_shape(self, name, shape):
        start = len(self.vect)
        size = np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, start + size), shape)
        self.vect = np.concatenate((self.vect, np.zeros(size)), axis=0)

    def new_vect(self, vect):
        assert vect.size == self.vect.size
        new_parser = self.empty_copy()
        new_parser.vect = vect
        return new_parser

    def empty_copy(self):
        """Creates a parser with a blank vector."""
        new_parser = VectorParser()
        new_parser.idxs_and_shapes = self.idxs_and_shapes.copy()
        new_parser.vect = None
        return new_parser

    def as_dict(self):
        return {k : self[k] for k in self.names}

    @property
    def names(self):
        return self.idxs_and_shapes.keys()

    def __getitem__(self, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(self.vect[idxs], shape)

    def __setitem__(self, name, val):
        if isinstance(val, list): val = np.array(val)
        if name not in self.idxs_and_shapes:
            self.add_shape(name, val.shape)

        idxs, shape = self.idxs_and_shapes[name]
        self.vect[idxs].reshape(shape)[:] = val

def fill_parser(parser, items):
    """Build a vector by assigning each block the corresponding value in
       the items vector."""
    partial_vects = [np.full(parser[name].size, items[i])
                     for i, name in enumerate(parser.names)]
    return np.concatenate(partial_vects, axis=0)

def make_nn_funs(layer_sizes):
    parser = VectorParser()
    for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        parser.add_shape(('weights', i), shape)
        parser.add_shape(('biases', i), (1, shape[1]))

    def predictions(W_vect, X):
        """Outputs normalized log-probabilities."""
        W = parser.new_vect(W_vect)
        cur_units = X
        N_iter = len(layer_sizes) - 1
        for i in range(N_iter):
            cur_W = W[('weights', i)]
            cur_B = W[('biases',  i)]
            cur_units = np.dot(cur_units, cur_W) + cur_B
            if i == (N_iter - 1):
                cur_units = cur_units - logsumexp(cur_units, axis=1)
            else:
                cur_units = np.tanh(cur_units)
        return cur_units

    def loss(W_vect, X, T, L2_reg=0.0):
        # TODO: consider treating L2_reg as a matrix
        log_prior = -np.dot(W_vect * L2_reg, W_vect)
        log_lik = np.sum(predictions(W_vect, X) * T) / X.shape[0]
        return - log_prior - log_lik

    return parser, predictions, loss


def construct_nn_multi(layer_sizes:list):
    
    parser = Weight_Parser()
    layers = zip(layer_sizes[:-1],layer_sizes[1:])
    m = len(layer_sizes)-1

    for i,l in enumerate(layers):
            parser.add(('mat',i),l)
            parser.add(('bias',i),l[1])

    def nn(w:np.ndarray, inputs:np.ndarray) -> float:
        
        for i in range(m):
            #print('input')
            #print(inputs)
            weight_matrix = parser.get(('mat',i),w)
            b = parser.get(('bias',i),w)
            outputs = np.dot(inputs,weight_matrix) + b
            inputs = np.tanh(outputs)
            #print('weights')
            #print(weight_matrix)
            #print('bias')
            #print(b)
            #print(f'outputs iter {i}:')
            #print(outputs)

        return inputs - logsumexp(inputs, axis=1) #, keepdims=True)
    
    def loss(w:np.ndarray, inputs:np.ndarray, targets:np.ndarray, L2_reg:float = 0 ):
        log_lik = np.sum(nn(w, inputs) * targets)/inputs.shape[0]
        #print('valore log lik')
        #print(log_lik)
        prior = np.dot(w * L2_reg,w)
        #print('valore prior')
        #print(prior)
        return -log_lik + prior


    return parser, nn, loss


def construct_nn_reg(layer_sizes:list):
    '''
    function to build neural network for L2 regularization
    '''
    parser = Weight_Parser()
    layers = zip(layer_sizes[:-1],layer_sizes[1:])
    m = len(layer_sizes)-1

    for i,l in enumerate(layers):
            parser.add(('mat',i),l)
            parser.add(('bias',i),l[1])

    def nn(w:np.ndarray, inputs:np.ndarray) -> float:
        
        for i in range(m):
            weight_matrix = parser.get(('mat',i),w)
            b = parser.get(('bias', i), w)
            b = 0.0
            outputs = np.dot(inputs,weight_matrix) + b
            print('input')
            print(inputs)
            print('weights')
            print(weight_matrix)
            print('bias')
            print(b)
            print(f'outputs iter {i}:')
            print(outputs)
            # Apply tanh activation to hidden layers only
            if i < m - 1:
                inputs = np.tanh(outputs)
            else:
                # Linear output layer
                inputs = outputs
            
        return inputs - logsumexp(inputs, axis=1, keepdims=True)

    
    def loss(w:np.ndarray, L2_reg: np.ndarray, inputs:np.ndarray, targets:np.ndarray):
        log_lik = np.sum(nn(w, inputs) * targets)/inputs.shape[0]
        print('valore log lik')
        print(log_lik)
        prior = np.dot(L2_reg * w, w)
        print('valore prior')
        print(prior)
        return -log_lik + prior #negative log likelihood + regularization prior


    return parser, nn, loss




#class WeightsParser(object):
#    def __init__(self):
#        self.idxs_and_shapes = {}
#        self.N = 0
#
#    def add_weights(self, name, shape):
#        start = self.N
#        self.N += np.prod(shape)
#        self.idxs_and_shapes[name] = (slice(start, self.N), shape)
#
#    def get(self, vect, name):
#        idxs, shape = self.idxs_and_shapes[name]
#        return np.reshape(vect[idxs], shape)
#
#def fill_parser(parser, items):
#    partial_vects = [np.full(parser[name].size, items[i])
#                     for i, name in enumerate(parser.names)]
#    return np.concatenate(partial_vects, axis=0)
#
#class BatchList(list):
#    def __init__(self, N_total, N_batch):
#        start = 0
#        while start < N_total:
#            self.append(slice(start, start + N_batch))
#            start += N_batch
#        self.all_idxs = slice(0, N_total)
#
#def logsumexp(X, axis):
#    max_X = np.max(X)
#    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))
#
#def make_nn_funs(layer_specs, L2_reg):
#
#    parser = WeightsParser()
#    for layer in layer_specs:
#        N_weights = layer.build_weights_dict()
#        parser.add_weights(layer, N_weights)
#
#    def predictions(W_vect, inputs):
#        """Outputs normalized log-probabilities."""
#        cur_units = inputs
#        for layer in layer_specs:
#            cur_weights = parser.get(W_vect, layer)
#            cur_units = layer.forward_pass(cur_units, cur_weights)
#        return cur_units - logsumexp(cur_units, axis=1)
#
#    def loss(W_vect, inputs, T):
#        log_prior = L2_reg * np.dot(W_vect, W_vect)
#        log_lik = np.sum(predictions(W_vect, inputs) * T) / inputs.shape[0]
#        return  -log_lik + log_prior
#
#    return parser, predictions, loss
#
#class MLP_layer:
#    def __init__(self, input_shape, out_shape) -> None:
#        self.input_shape = input_shape
#        self.out_shape = out_shape
#
#    def forward_pass(self, inputs, param_vector):
#        weights = self.parser.get(param_vector, "weights")
#        biases = self.parser.get(param_vector, "biases")
#        
#        return self.nonlinearity(np.dot(inputs, weights) + biases)
#
#    def build_weights_dict(self):
#        
#        self.parser = WeightsParser()
#        self.parser.add_weights("weights",(self.input_shape, self.out_shape))
#        self.parser.add_weights("biases", (self.out_shape,))
#    
#        return self.parser.N
#
#class tanh_layer(MLP_layer):
#    def nonlinearity(self, x):
#        return np.tanh(x)
#