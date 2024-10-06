import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import pickle
import math

# ALL EQUATIONS USED AND THEIR DERIVATIVES
class Equation:
    def __init__(self):
        raise ReferenceError('An object can not exist for this class')
    
    # ========================
    # ACTIVATION FUNCTIONS
    # ========================
    
    def none(x, dx=False):
        return x if not dx else x ** 0
        
    def sigmoid(x, dx=False):
        return 1 / (1 + np.exp(-x)) if not dx else (np.exp(-x)) / ((1 + (np.exp(-x))) ** 2)
    
    def relu(x, dx=False):
        return np.maximum(0, x) if not dx else np.where(x > 0, 1, 0)
    
    def leaky_relu(x, dx=False):
        alpha = 0.01
        return np.maximum(alpha * x, x) if not dx else np.where(x > 0, 1, alpha)
        
    def tanh(x, dx=False):
        return np.tanh(x) if not dx else 1 - (np.tanh(x) ** 2)
    
    def elu(x, dx=False):
        return np.where(x > 0, x, np.exp(x) - 1) if not dx else np.where(x > 0, 1, np.exp(x))
    
    def softmax(x, dx=False):
        smout = np.exp(x) / sum(np.exp(x))
        if np.isnan(smout.any()):
            print(smout, np.exp(x), sum(np.exp(x)))
        return smout if not dx else smout * (1 - smout)
            
    def step(x, dx=False):
        return np.where(x > 0, 1, 0) if not dx else 0
    
    # ========================
    # LOSS FUNCTIONS
    # ========================
    
    def mse(y_pred, y_actu, dx=False):
        # MEAN SQUARED ERROR
        return np.square(y_pred - y_actu) / 2 if not dx else y_pred - y_actu
       
    def bce(y_pred, y_actu, dx=False):
        # BINARY CROSS-ENTROPY LOSS
        epsilon = 1e-10
        return -(y_actu * np.log(y_pred + epsilon)) - ((1 - y_actu) * np.log(1 - y_pred + epsilon)) if not dx else -(y_actu / (y_pred + epsilon)) + ((1 - y_actu) / (1 - y_pred + epsilon))
        
    def cce(y_pred, y_actu, dx=False):
        # CATEGORIAL CROSS-ENTROPY LOSS
        epsilon = 1e-10
        #print(f'CCE :   {np.log(y_pred + epsilon), y_actu / (y_pred + epsilon)}')
        return -np.mean(y_actu * np.log(y_pred + epsilon), axis=0) if not dx else -np.mean(y_actu / (y_pred + epsilon), axis=0)
        
    # ========================
    # NORMALIZATION EQUATIONS
    # ========================
    
    def n_minmax(dataset: np.ndarray, min: float, max: float, inverse=False):
        if dataset is not None:
            return (dataset - min) / (max - min) if not inverse else ((dataset * (max - min)) + min)
        return None
    
    def n_zscore(dataset: np.ndarray, mean: float, stddev: float, inverse=False):
        if dataset is not None:
            return (dataset - mean) / stddev if not inverse else (dataset * stddev) + mean
        return None
    


class Layer:
    
# ------------------------
# VALUES FOR EACH LAYER, ORGANIZATION
# ------------------------

    def __init__(self, nodes: int, activation=None):
        # number of nodes in the layer
        self.nodes = nodes
        # activation function for the layer, NONE if the layer is an input layer
        self.activation = activation
        
    def __repr__(self) -> str:
        return f'<{self.nodes} nodes | {self.activation}>'

# NETWORK CLASS, TIES ALL NEURONS + INPUTS + OUTPUTS together
class FCNN:
    
# ------------------------
# NETWORK ASSEMBLY
# ------------------------

    def __init__(self, layers: list, loss_func):
        # list containing layer objs for each layer and easy retrival
        self.layers = layers
        # std and mean for inputs, conforms to normalization
        self.normdata = {}
        # set the loss of the network, used to check for best epoch
        self.loss = None
        # loss function
        self.loss_func = loss_func

    def __repr__(self) -> str:
        # flip through all inputs, hidden nodes, outputs
        string = ''
        for layer in self.layers:
            string += layer + '\n'

    # create and store connections between all inputs, hidden neurons, outputs
    def generate_params(self, weight_init: str = 'xaviar', bias_init: str = 'small-normal', rounding_place: int = None) -> None:
        # WEIGHT MATRICES - layer > node > node connections to next layer
        if weight_init == 'xaviar':
            self.weight_matrices = [np.array([np.random.normal(0.0, np.sqrt(2 / (layer.nodes + self.layers[i+1].nodes)), size=self.layers[i+1].nodes) for node in range(layer.nodes)]) for i, layer in enumerate(self.layers) if i+1 < len(self.layers)]
        elif weight_init == 'he':
            self.weight_matrices = [np.array([np.random.normal(0, np.sqrt(6 / layer.nodes), size=self.layers[i+1].nodes) for node in range(layer.nodes)]) for i, layer in enumerate(self.layers) if i+1 < len(self.layers)]
        else:
            raise ValueError('Invalid value for weight initialization')
        
        # BIAS VECTORS - layer > node
        if bias_init == 'zeros':
            self.bias_vectors = [np.random.zeros(layer.nodes) for layer in self.layers[1:]]
        elif bias_init == 'small-normal':
            self.bias_vectors = [np.random.randn(layer.nodes) * 0.01 for layer in self.layers[1:]]
        elif bias_init == 'normal':
            self.bias_vectors = [np.random.randn(layer.nodes) for layer in self.layers[1:]]
        else:
            raise ValueError('Invalid value for bias initialization')
        
        # ANY ROUNDING FOR DEBUG OR OTHER
        if rounding_place is not None:
            for i, param_tuple in enumerate(zip(self.weight_matrices, self.bias_vectors)):
                self.weight_matrices[i] = np.around(param_tuple[0], rounding_place)
                self.bias_vectors[i] = np.around(param_tuple[1], rounding_place)

# ------------------------
# INPUT PREPROCESSING
# ------------------------
    
    def fit_norm(self, trainx: np.ndarray, trainy: np.ndarray, norm_func) -> None:
        # check for the norm function passed, then assign the variables needed to execute that norm function
        if norm_func == Equation.n_zscore:
            self.normdata['mean-x'], self.normdata['stddev-x'] = np.mean(trainx), np.std(trainx)
            self.normdata['mean-y'], self.normdata['stddev-y'] = np.mean(trainy), np.std(trainy)
        elif norm_func == Equation.n_minmax:
            self.normdata['min-x'], self.normdata['max-x'] = np.amin(trainx), np.amax(trainx)
            self.normdata['min-y'], self.normdata['max-y'] = np.amin(trainy), np.amax(trainy)
            
    def undo_norm(self, y) -> tuple:
        if np.isin(np.array(['mean-x', 'mean-y', 'stddev-x', 'stddev-y']), list(self.normdata.keys())).all():
            return Equation.n_zscore(y, self.normdata['mean-y'], self.normdata['stddev-y'], inverse=True)
        elif np.isin(np.array(['min-x', 'min-y', 'max-x', 'max-y']), list(self.normdata.keys())).all():
            return Equation.n_zscore(y, self.normdata['min-y'], self.normdata['max-y'], inverse=True)
        else:
            return y

    def norm(self, x: np.ndarray = None, y: np.ndarray = None) -> tuple:
        if np.isin(np.array(['mean-x', 'mean-y', 'stddev-x', 'stddev-y']), list(self.normdata.keys())).all():
            return None if x is None else Equation.n_zscore(x, self.normdata['mean-x'], self.normdata['stddev-x']), None if y is None else Equation.n_zscore(y, self.normdata['mean-y'], self.normdata['stddev-y'])
        elif np.isin(np.array(['min-x', 'min-y', 'max-x', 'max-y']), list(self.normdata.keys())).all():
            return None if x is None else Equation.n_minmax(x, self.normdata['min-x'], self.normdata['max-x']), None if y is None else Equation.n_zscore(y, self.normdata['min-y'], self.normdata['max-y'])
        return (x, y)
    
    @classmethod    
    def tt_divide(self, x: np.ndarray, y: np.ndarray, training_portion: float = 0.8, testing_portion: float = 0.2) -> tuple:
        if training_portion + testing_portion != 1.0:
            raise ValueError('TTP does not sum to one.')
        
        perm = np.random.permutation(len(x))
        #shuffled_in = input_dpt[perm]
        #shuffled_out = output_dpt[perm]
        shuffled_in = x
        shuffled_out = y
        partition = int(np.floor(len(x) * training_portion))
        return shuffled_in[:partition], shuffled_out[:partition], shuffled_in[partition:], shuffled_out[partition:]

    @classmethod
    def batch_div(self, trainx: np.ndarray, trainy: np.ndarray, batch_size: int) -> tuple:
        batch_count = int(np.floor(len(trainx) / batch_size))
        xbatches = np.array([trainx[i*batch_size:(i+1)*batch_size][0] for i in range(batch_count)])
        ybatches = np.array([trainy[i*batch_size:(i+1)*batch_size][0] for i in range(batch_count)])
        return xbatches, ybatches
    
    def prepdata(self, x: np.ndarray, y: np.ndarray, training_portion: float = 0.8, testing_portion: float = 0.2, batch_size: int = 1, norm_func = None):
        trainx, trainy, testx, testy = FCNN.tt_divide(x, y, training_portion, testing_portion)
        if norm_func is not None:
            self.fit_norm(trainx, trainy, norm_func)
            trainx = self.norm(x=trainx)[0]
            testx = self.norm(x=testx)[0]
            pass
        trainx, trainy = FCNN.batch_div(trainx, trainy, batch_size)
        return trainx, trainy, testx, testy

# ------------------------
# CLASS METHODS / SAVING
# ------------------------

    @classmethod
    def save_net(self, net: object, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(net, file)
     
    @classmethod
    def load_net(self, path: str) -> object:
        with open(path, 'rb') as file:
            return pickle.load(file) 

# ------------------------
# PROCESSING / PROPAGATION
# ------------------------
    
    # RUN A GIVEN VALUE THROUGH THE NETWORK
    def predict(self, x: np.ndarray) -> np.ndarray:
        
        layer_vals = np.array(x)
        # iterate through the layers via weight matrices for ease
        for i, layer in enumerate(self.layers[1:]):
            layer_vals = layer.activation(np.dot(np.insert(layer_vals, 0, 1, axis=-1), np.insert(self.weight_matrices[i], 0, self.bias_vectors[i], axis=0)))
           
        return layer_vals
    
    

    def backprop(self, trainx: np.ndarray, trainy: np.ndarray, testx: np.ndarray, testy: np.ndarray, epochs: int, lr: float) -> None:
        
        if len(trainx) != len(trainy):
            raise ValueError(f'Input and output data are unequal ({len(trainx)} : {len(trainy)})')
        
        for epoch in range(epochs):
            for x, y in zip(trainx, trainy):
                print(x, y)
                x, y = x[0], y[0]
                
                # ========================
                # FORWARD PASS
                # ======================== 
                
                # for use during backwards pass
                weighted_sums = [x]
                activated_sums = [x]
                
                # iterate through the layers
                for i, layer in enumerate(self.layers[1:]):
                    z = np.dot(np.insert(activated_sums[i], 0, 1, axis=-1), np.insert(self.weight_matrices[i], 0, self.bias_vectors[i], axis=0))
                    weighted_sums.append(z)
                    activated_sums.append(layer.activation(z))
                
                # ========================
                # BACKWARD PASS (OUTPUT LAYER)
                # ======================== 
                
                # gather all necessary gradients
                dx_loss = self.loss_func(y_pred=activated_sums[-1], y_actu=y, dx=True)
                dx_out_actisum = self.layers[-1].activation(weighted_sums[-1], dx=True)
                
                # error term for the first hidden layer
                eterms = np.array([dx_loss * dx_out_actisum])
                
                # parameter updates for weights and biases
                wei_eterm = dx_out_actisum.T.dot(dx_loss) / int(x.size)
                self.weight_matrices[-1] -= wei_eterm * lr
                self.bias_vectors[-1] -= np.mean(dx_loss, axis=0, keepdims=True) * lr
                
                # ========================
                # BACKWARD PASS (HIDDEN LAYERS)
                # ========================
                
                for k, layer in enumerate(self.layers[:-1][::-1]):
                    
                    dx_out_weightedsum = self.weight_matrices[-1]

# ------------------------
# POST-PROCESSING
# ------------------------  
        
    def cost(self, testx: np.ndarray, testy: np.ndarray):
        return np.mean(self.loss_func(self.predict(testx), testy))
        
    def test_efficiency(self, base_passes: int, repetitions: int) -> dict:
        times = []
        for i in range(repetitions):
            start_time = time.time()
            for j in range(base_passes):
                self.predict(np.array([np.random.randint(-5, 5) for k in range(self.neuron_layers[0])]))
            times.append(time.time() - start_time)
        return {'avg': sum(times) / len(times), 'sum': sum(times)}
        
    def graph1_1(self, input_dtp: np.ndarray, output_dtp: np.ndarray) -> None:
        odtp_pred = np.array([self.predict(self.norm(x=i)[0]) for i in input_dtp]) 
        
        plt.plot(input_dtp, output_dtp, color='red', label='Actual')
        plt.plot(input_dtp, odtp_pred, color='blue', label='Predicted')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('FNN Actual vs. Predicted')
        plt.grid(True)
        plt.show()    

    def eval_metrics(self, inps: np.ndarray, outs: np.ndarray, metrics = []):
        def loss(x: np.ndarray, y: np.ndarray):
            return self.calc_loss(x, y)
        
        def accuracy(x: np.ndarray, y: np.ndarray):
                if len(y.shape) != 1:
                    y = np.argmax(y, axis=-1)
                return np.mean(np.argmax(self.predict(x), axis=-1) == y)
        
        def mae(x: np.ndarray, y: np.ndarray):
            return np.mean(np.abs(y - self.predict(x)))
        
        def mse(x: np.ndarray, y: np.ndarray):
            return np.mean(np.power(y - self.predict(x), 2))
        
        return {metric: {'loss': loss, 'accuracy': accuracy, 'mae': mae, 'mse': mse}[metric](inps, outs) for metric in metrics}
    

# ------------------------
# END OF FNN FRAMEWORK
# ------------------------  
                  

'''# only for dataset
import tensorflow as tf
                  
net = net = Network([Layer(784), Layer(360, Equation.sigmoid), Layer(120, Equation.sigmoid), Layer(40, Equation.sigmoid), Layer(10, Equation.softmax)], Equation.cce)
net.generate_matrices(weight_init='xaviar')

# FORMATTING AND LOADING MNIST
mnist = tf.keras.datasets.mnist
(train_pix, train_lab), (test_pix, test_lab) = mnist.load_data()

pix = np.concatenate((train_pix / 255, test_pix / 255))
pix = pix.reshape(pix.shape[0], -1)
lab = np.concatenate((np.array([np.array([1 if i == digit else 0 for i in range(10)]) for digit in train_lab]), np.array([np.array([1 if i == digit else 0 for i in range(10)]) for digit in test_lab])))
#lab = np.concatenate((train_lab, test_lab))

# TRAINING AND DATA
train_pix, train_lab, test_pix, test_lab = net.prepdata(pix, lab, 0.8, 0.2, 64, None)

print(f'Training Start ({net.eval_metrics(test_pix, test_lab, ["accuracy", "mae"])})')
net.backprop(train_pix, train_lab, test_pix, test_lab, num_epochs=20, learning_rate=0.000001, autosave_path="mnist_jetsave.pickle", max_grad=1.0, ge_persistance=10)
print((net.eval_metrics(test_pix, test_lab, ['accuracy', 'mae'])))

Network.save_net(net, "mnist_jetsave.pickle")

# met = Network.load_net("mnist_netsave.pickle")'''

x = np.array([[15, 10], [15, 10], [15, 10], [15, 10], [15, 10], [15, 20], [15, 20], [15, 20], [15, 20], [15, 20], [15, 30], [15, 30], [15, 30], [15, 30], [15, 30], [15, 40], [15, 40], [15, 40], [15, 40], [15, 40], [15, 50], [15, 50], [15, 50], [15, 50], [15, 50], 
          [30, 10], [30, 10], [30, 10], [30, 10], [30, 10], [30, 20], [30, 20], [30, 20], [30, 20], [30, 20], [30, 30], [30, 30], [30, 30], [30, 30], [30, 30], [30, 40], [30, 40], [30, 40], [30, 40], [30, 40], [30, 50], [30, 50], [30, 50], [30, 50], [30, 50],
          [45, 10], [45, 10], [45, 10], [45, 10], [45, 10], [45, 20], [45, 20], [45, 20], [45, 20], [45, 20], [45, 30], [45, 30], [45, 30], [45, 30], [45, 30], [45, 40], [45, 40], [45, 40], [45, 40], [45, 40], [45, 50], [45, 50], [45, 50], [45, 50], [45, 50],
          [60, 10], [60, 10], [60, 10], [60, 10], [60, 10], [60, 20], [60, 20], [60, 20], [60, 20], [60, 20], [60, 30], [60, 30], [60, 30], [60, 30], [60, 30], [60, 40], [60, 40], [60, 40], [60, 40], [60, 40], [60, 50], [60, 50], [60, 50], [60, 50], [60, 50]])
y = np.array([[5.68], [5], [5.88], [4.72], [5.28], [6.03], [6.06], [5.69], [6], [5.78], [7.35], [7.43], [7.32], [7.25], [6.79], [8.63], [8.78], [8.82], [8], [8.84], [10.03], [10.69], [10.9], [9.65], [11.69],
        [5.63], [5.54], [5.31], [5.41], [5.28], [5.88], [6.53], [6.46], [6.01], [6.22], [7.1], [7.19], [6.92], [6.97], [7.16], [8.13], [8.93], [8.97], [8.34], [9.25], [11.22], [11.16], [10.5], [11.28], [11.12],
        [5.41], [5.53], [5.69], [5.65], [5.53], [6.4], [6.38], [5.78], [5.88], [6.93], [8.03], [7.59], [7.6], [8.06], [8.1], [9.53], [9.69], [9.69], [9.94], [10.16], [12.94], [12.28], [12.88], [13], [12.77], 
        [6.09], [6.03], [5.53], [6.18], [5.56], [7.28], [7], [7.18], [6.56], [6.75], [7.81], [8.31], [8.65], [8.84], [8.78], [11.16], [10.06], [10.63], [10.94], [11.15], [13.06], [13.07], [13.87], [13.5], [14.25]])

met = FCNN([Layer(2), Layer(4, Equation.elu), Layer(9, Equation.elu), Layer(1, Equation.none)], loss_func=Equation.mse)
met.generate_params()

trainx, trainy, testx, testy = met.prepdata(x, y, batch_size=8)
met.backprop(trainx, trainy, testx, testy, epochs=100, lr=0.0001)


'''inps = np.array([np.array([x]) for x in np.array(range(500)) / 500])
outs = np.array([np.array([2*x]) for x in np.array(range(500)) / 500])

met = FCNN([Layer(1), Layer(1, Equation.none)], loss_func=Equation.mse)
met.generate_params(weight_init='xaviar', bias_init='small-normal')

trainx, trainy, testx, testy = met.prepdata(inps, outs, 0.8, 0.2, 1, None)

print(met.weight_matrices, met.bias_vectors)
met.backprop(trainx, trainy, testx, testy, epochs=200, lr=0.0001)
print(met.weight_matrices, met.bias_vectors)

met.graph1_1(inps, outs)
print(met.eval_metrics(inps, outs, ['mae']))'''