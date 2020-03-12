import numpy as np
from scipy.special import expit as expit
class NeuralNetwork():
    '''
    Neural Net class to be used throughout project
    '''
    def __init__(self, input, test_set = None, num_hidden = 3, output_dim = 8, learn_rate = 0.01, reg_term = 0.00001, rounds = 100000, bias=1, expected_out = None):
        '''
        Initialize NeuralNetwork object
        Input: training input, number of nodes in hidden layer, output dimensions, learning rate, regularization term, rounds of gradient descent, bias, expected output
        '''

        if expected_out == 'auto':
            self.exp = input
            self.l0 = input
            self.input_dim = len(input[0])
        elif expected_out == 'blind':
            self.exp = input[:,:1]
            self.l0 = input[:, 1:]
            self.input_dim = len(input[0]) - 1
            self.test_set = test_set
            self.test_exp = None
        else:
            self.exp = input[:,:1]
            self.l0 = input[:, 1:]
            self.input_dim = len(input[0]) - 1
            self.test_exp = test_set[:,:1]
            self.test_set = test_set[:, 1:]

        self.num_hidden = num_hidden
        self.output_dim = output_dim
        self.b = bias
        self.lr = learn_rate
        self.reg = reg_term
        self.rounds = rounds


        self.l1 = np.ones(self.num_hidden)
        self.l2 = np.ones(self.output_dim)

        self.w1, self.w2 = self.make_weights()

        self.model = {}





    def make_weights(self):
        '''
        Initial weights prior to first pass in the NeuralNetwork
        Input: NN object
        Output: 2 vectors of weights
        '''
        w1 = 2*np.random.randn(self.input_dim, self.num_hidden) - 1
        w2 = 2*np.random.randn(self.num_hidden, self.output_dim) - 1
        return w1, w2




    def feed_forward(self, train_in):
        '''
        Feed forward propagation through the 3 level neural NeuralNetwork
        Input: called on NN object
        Output: output layer
        '''
        self.l1 = activation(x=np.dot(self.l0, self.w1), dx = False) # hidden layer
        self.l2 = activation(x = np.dot(self.l1, self.w2), dx = False) # output layer

        return self.l2

    def backprop(self):
        '''
        backpropagation for gradient descent and updating make_weights
        Input: called on NN object
        Output: error
        '''
        l2_error = self.exp - self.l2
        l2_change = l2_error*activation(x = self.l2, dx = False)

        #backpropogation for gradient descent
        l1_error = l2_change.dot(self.w2.T)
        l1_change = l1_error*activation(x = self.l1, dx = True)


        self.w1 += self.lr * (self.l0.T.dot(l1_change) + self.reg*self.w1)
        self.w2 += self.lr * (self.l1.T.dot(l2_change) + self.reg*self.w2)

        return l2_error

    def fit(self):
        '''
        Inspiration/assistance from Taylor Cavasos (I sit near her in lab right now)
        Fitting the encoder/decode model for the neural net
        Input: called on NN object
        Output: errors
        '''
        MSE = [float("inf")]*(self.rounds)
        i = 0
        while i < self.rounds and not (i < 2 and (round(MSE[i-1],6) >= round(MSE[i-2],6) and MSE[i-1] < .00001)):
            self.l2 = self.feed_forward(self.l0)
            error = self.backprop()
            MSE[i] = np.average(np.square(error))
            i += 1

        return [x for x in MSE if x != float("inf")]


    def predict(self):
        '''
        Use trained model to predict on new set
        Input: test_set
        Output: predictions
        '''

        l1 = activation(np.dot(self.test_set, self.w1))
        l2 = activation(np.dot(l1, self.w2))

        if self.test_exp is not None:
            err = self.test_exp - l2
            mse = np.average(np.square(err))
        else:
            mse = None
        return l2, mse

def activation(x, dx = False):
    '''
    Sigmiod based activation function - as discussed by Mike Keiser in class
    Input: value to be 'activated' can be int, float or array, boolean of if we want the derivative or not
    Output: sigmoid activation of input, derivative if requested
    using expit from scipy to prevent runtime overflow errors
    '''
    sig = expit(x + 1)
    if dx:
        return sig*(1-sig)
    else:
        return sig
