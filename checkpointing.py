#import autograd.numpy as np #use this if hyper_grad_lr is defined here
import numpy as np
from enum import IntEnum
from abc import ABCMeta


"""
In the following I try to implement Hyper Grad without using the exact representation but
by taking the forward steps (training the parameters) multiple times and saving the checkpoints
"""

"""
For each interval actually under consideration, revolve
calculates an appropriate 't' called reps to determine the state to which the
next checkpoint will be set
"""
checkup = 10
repsup = 64


def numforw(steps, snaps):
    """
    return the number of forward steps from steps and num
    """
    if snaps < 1:
        print("Error: snaps < 1")
        return -1

    if snaps > checkup:
        print(f"Error: snaps ({snaps}) greater than checkup ({checkup})")
        return -1

    reps = 0
    range_ = 1
    while range_ < steps:
        reps += 1
        range_ = range_ * (reps + snaps) // reps

    print(f"range = {range_}, reps = {reps}")
    if reps > repsup:
        print(f"Error: reps ({reps}) greater than repsup ({repsup})")
        return -1

    num = reps * steps - range_ * reps // (snaps + 1)
    return num

def maxrange(ss, tt):
    
    
    if tt < 0 or ss < 0:
        print("Error in MAXRANGE: negative parameter")
        return -1

    res = 1.0 

    for i in range(1, tt + 1):  # `range(1, tt+1)` include `tt`
        res *= (ss + i)
        res /= i
 
    #ires = beta(ss, tt)
    ires = int(res) 
    return ires

def adjust(n_iter):
    ''' 
    function to get an approximated optimal number of checkpoints to reverse 
    '''
    snaps = 1
    reps = 1
    s = 0

    # Primo ciclo: ridurre `s` finché maxrange supera `n_iter`
    #while beta(snaps + s, reps + s)>n_iter
    #questo while controlla se beta(1,1) > n_iter, beta(1,1) = 2.
    while maxrange(snaps + s, reps + s) > n_iter:
        s -= 1

    # Secondo ciclo: aumentare `s` finché maxrange è inferiore a `n_iter`
    while maxrange(snaps + s, reps + s) < n_iter:
        s += 1

    # Aggiorna snaps e reps
    snaps += s
    reps += s
    #in questo momento snaps = reps = s+1
    s = -1
    
    # Riduci snaps o reps finché maxrange è maggiore o uguale a `n_iter`
    while maxrange(snaps, reps) >= n_iter:
        #qua si riducono in maniera alternata snaps e reps, perché si parte con snaps = reps e 
        #poi snaps= reps +1 e poi di nuovo snaps = reps
        if snaps > reps:
            snaps -= 1
            s = 0
        else:
            reps -= 1
            s = 1

    # Correzione finale per snaps o reps
    if s == 0:
        snaps += 1
    elif s == 1:
        reps += 1

    return snaps

class ActionType(IntEnum):
   
    advance = 0
    takeshot = 1
    restore = 2
    firsturn = 3
    youturn = 4
    terminate = 5
    error = 6

class Checkpoint:
    def __init__(self, snaps):
        """
        Inizializza l'oggetto Checkpoint.
        Args:
            snaps (int): Numero di snapshot da gestire.
        """
        self.ch = [0] * snaps
        self.number_of_reads = [0] * snaps
        self.number_of_writes = [0] * snaps
        self.takeshots = 0
        self.advances = 0
        self.commands = 0

class BinomialCKP():
    '''
    scheduler class to get action to take
    self.check : number of checkpoints currently assigned
    self.checkpoint.ch : ch list that indicates positions of assigned checkpoints
    self.capo : position indicating the start of the subrange you are currently trying to reverse
    self.fine : position indicating the end of the subrange you are currently trying to reverse
    self.snaps : number of checkpoints you want to use with limit self.checkup

    '''    
    def __init__(self, st, f=None):

        self.snaps = adjust(st)
        self.checkpoint = Checkpoint(self.snaps)    
        self.steps = st
        self.check = -1
        self.info = 3
        self.fine = self.steps
        self.capo = 0
        self.oldcapo = 0
        self.checkup = 100
        self.repsup = 1000
        self.info = 3
        self.oldsnaps = self.snaps

    def revolve(self):
        """
        revolve scheduler to get action to execute at a given point
        """
        self.checkpoint.commands += 1
        

        if self.check < -1 or self.capo > self.fine:
            print('capo > fine')
            print(self.capo)
            print(self.fine)
            return ActionType.error

        if self.check == -1 and self.capo < self.fine:
            self.turn = 0  # Inizializzazione del contatore di turn
            self.checkpoint.ch[0] = self.capo - 1

        diff = self.fine - self.capo
        if diff == 0:  # Riduci capo al checkpoint precedente
            if self.check == -1 or self.capo == self.checkpoint.ch[0]:
                if self.info > 0:
                    print(f"\n advances: {self.checkpoint.advances:5}")
                    print(f" takeshots: {self.checkpoint.takeshots:5}")
                    print(f" commands: {self.checkpoint.commands:5}\n")
                return ActionType.terminate
            else:

                self.capo = self.checkpoint.ch[self.check] #capo diventa il numero che c'è dentro all'ultimo checkpoint
                self.oldfine = self.fine
                self.checkpoint.number_of_reads[self.check] += 1
                return ActionType.restore

        elif diff == 1:  # passo combinato forward/reverse
            self.fine -= 1
            if self.check >= 0 and self.checkpoint.ch[self.check] == self.capo:
                self.check -= 1
            #controllo se è il primo passo di reverse 
            if self.turn == 0:
                self.turn = 1
                self.oldfine = self.fine
                return ActionType.firsturn
            else:
                self.oldfine = self.fine
                return ActionType.youturn

        else:
            if self.check == -1:  # Inizializzazione
                self.checkpoint.ch[0] = 0
                self.check = 0
                self.oldsnaps = self.snaps
                if self.snaps > self.checkup:
                    self.info = 14
                    print(14)
                    return ActionType.error

                if self.info > 0:
                    num = numforw(self.fine - self.capo, self.snaps)
                    if num == -1:
                        self.info = 12
                        print(12)
                        return ActionType.error
                    print(f" prediction of needed forward steps: {num:8}")
                    print(f" slowdown factor: {num / (self.fine - self.capo):.4f}\n")

                self.oldfine = self.fine
                self.checkpoint.number_of_writes[self.check] += 1
                self.checkpoint.takeshots += 1
                return ActionType.takeshot

            if self.checkpoint.ch[self.check] != self.capo:  # Takeshot

                self.check += 1 #aggiungi un checkpoint
                if self.check >= self.checkup or self.check + 1 > self.snaps:
                    self.info = 10 if self.check >= self.checkup else 11
                    print(10)
                    return ActionType.error

                self.checkpoint.ch[self.check] = self.capo
                self.checkpoint.takeshots += 1
                self.oldfine = self.fine
                self.checkpoint.number_of_writes[self.check] += 1
                return ActionType.takeshot

            else:  # Advance
                
                if self.oldfine < self.fine and self.snaps == self.check + 1:
                    self.info = 13
                    print(13)
                    return ActionType.error

                self.oldcapo = self.capo #add checkpoints to the list
                ds = self.snaps - self.check #numero di checkpoint che posso ancora aggiungere                

                if ds < 1:
                    self.info = 11
                    return ActionType.error

                reps = 0
                range_ = 1
                while range_ < self.fine - self.capo:
                    reps += 1
                    range_ = range_ * (reps + ds) // reps
                
                #range is equal to beta(ds,reps) where beta is binomial coefficient (ds+reps, reps)

                if reps > self.repsup:
                    self.info = 15
                    return ActionType.error

                if self.snaps != self.oldsnaps and self.snaps > self.checkup:
                    self.info = 14
                    return ActionType.error

                bino1 = range_ * reps // (ds + reps) # bino1 = beta(ds, reps-1)
                bino2 = (bino1 * ds) // (ds + reps - 1) if ds > 1 else 1 #bino2 = beta(ds-1, reps-1)
                bino3 = bino2 * (ds - 1) // (ds + reps - 2) if ds > 2 else (0 if ds == 1 else 1) # bino3 = beta(ds-2, reps-1)
                bino4 = bino2 * (reps - 1) // ds #bino4 = beta(ds, reps-2)
                bino5 = bino3 * (ds - 2) // reps if ds > 3 else (0 if ds < 3 else 1) #bino5 = beta(ds-3, reps)

                #update capo following rule at the end of page 34 of revolve paper by Griewank and Walther
                if self.fine - self.capo <= bino1 + bino3:
                    self.capo += bino4
                elif self.fine - self.capo >= range_ - bino5:
                    self.capo += bino1
                else:
                    self.capo = self.fine - bino2 - bino3

                if self.capo == self.oldcapo:
                    self.capo = self.oldcapo + 1

                self.checkpoint.advances += self.capo - self.oldcapo
                self.oldfine = self.fine
                return ActionType.advance

from hyperoptimizer import load_alpha
from autograd import grad



def hyper_grad_lr(nSteps, parser, loss, f, multi_alpha, multi_gamma, w_0, v_0, batch_idxs, num_epochs ):

    scheduler = BinomialCKP(nSteps)
    stack = [{} for _ in range(scheduler.snaps)]
    iters = list(zip(range(nSteps), multi_alpha, multi_gamma, batch_idxs * num_epochs))
    gradient = grad(loss)
    l_grad   = grad(f)
    w, v = w_0, v_0


    def forward(check:int, nfrom: int, nto: int):

        w = stack[check]['weights']
        v = stack[check]['velocity']

        for t, alpha, gamma, batch in iters[nfrom:nto]:
            print(f'forward step number {t}')
            cur_alpha = load_alpha(parser, alpha)
            cur_gamma = load_alpha(parser, gamma)
            g =  gradient(w, batch)
            v *= cur_gamma
            v -= (1 - cur_gamma)*g
            w += cur_alpha*v

        return w, v

    def reverse(iteration, w, v, d_w, d_v, d_alpha, d_gamma):
        '''
        This function does only one step of RMD
        '''
        hyper_gradient = grad(lambda w, idx, d: np.dot(gradient(w,idx),d))
        i, alpha, gamma, batch = iters[iteration]
        print(f'backprop step {i}')
        cur_alpha = load_alpha(parser, alpha)
        cur_gamma = load_alpha(parser, gamma)
        for j, (ixs, _) in enumerate(parser.shape_idx.values()):
                d_alpha[i,j] = np.dot(d_w[ixs], v[ixs])

        #gradient descent reversion
        g  = gradient(w, batch)
        w -= cur_alpha * v
        v += (1-cur_gamma)*g
        v /= cur_gamma

        d_v += cur_alpha*d_w
        for j, (ixs, _) in enumerate(parser.shape_idx.values()):
                d_gamma[i,j] = np.dot(d_v[ixs], v[ixs] + g[ixs])
        print('get dw')
        d_w -= (1-cur_gamma)*hyper_gradient(w, batch, d_v)
        print('done')
        d_v *= cur_gamma

        return d_w, d_v, d_alpha, d_gamma

    
    while(True):
        action = scheduler.revolve()
        print(action)
        if action == ActionType.advance:
            print(f'advance the system from {scheduler.oldcapo} to {scheduler.capo}')
            w, v = forward(scheduler.check, scheduler.oldcapo, scheduler.capo)
        elif action == ActionType.takeshot:
            print('saving current state')
            print(scheduler.check)
            stack[scheduler.check]['weights']  = w
            stack[scheduler.check]['velocity'] = v
        elif action == ActionType.firsturn:
            print('executing first reverse step')
            wF, vF = forward(scheduler.check, scheduler.oldcapo, nSteps)
            #initialise gradient values
            d_alpha, d_gamma = np.zeros(multi_alpha.shape), np.zeros(multi_gamma.shape)
            d_v = np.zeros_like(w_0)
            d_w = l_grad(wF, batch_idxs.all_idxs)  
            #first step
            d_w, d_v, d_alpha, d_gamma = reverse(nSteps-1, wF, vF, d_w, d_v, d_alpha, d_gamma)
        elif action == ActionType.restore:
            print(f'loading state number {scheduler.check}')
            w, v = stack[scheduler.check]['weights'], stack[scheduler.check]['velocity']
        elif action == ActionType.youturn:
            print(f' doing reverse step at time {scheduler.fine}')
            d_w, d_v, d_alpha, d_gamma = reverse(scheduler.fine, w, v, d_w, d_v, d_alpha, d_gamma)
        if action == ActionType.terminate:
            break

    return d_w, d_v, d_alpha, d_gamma