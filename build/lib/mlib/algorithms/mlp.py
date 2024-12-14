import numpy as np
import matplotlib.pyplot as plt
import pickle


# --- # --- # --- # --- # --- # --- #


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
class MultilayerPerceptron:
   
# ------------------------
# NETWORK ASSEMBLY
# ------------------------


    def __init__(self, *layers, loss_func):
        # list containing layer objs for each layer and easy retrival
        self.layers = layers
        # std and mean for inputs, conforms to normalization
        self.normdata = {}
        # set the loss of the network, used to check for best epoch
        self.loss = None
        # loss function
        self.loss_func = loss_func
        # define the params (w / b)
        self.weight_matrices = None
        self.bias_vectors = None
        # saves epoch metric data for training / testing
        self.metric_data = []


    def __repr__(self) -> str:
        # flip through all inputs, hidden nodes, outputs
        string = ''
        for i, layer in enumerate(self.layers):
            string += '\033[0m{str(layer)}{("\n\t\033[90mweights:\n" + str(self.weight_matrices[i])) if i != len(self.layers)-1 and self.weight_matrices is not None else ""}{("\n\t\033[90mbiases:\n" + str(self.bias_vectors[i-1])) if i != 0 and self.bias_vectors is not None else ""}\n'
        return string + '\033[0m'


    # create and store connections between all inputs, hidden neurons, outputs
    def generate_params(self, weight_init: str = 'xaviar', bias_init: str = 'small-normal', rounding_place: int = None) -> None:
        """
        Generate the weights and biases of the network using the specified methods


        Args:
            weight_init (str): The weight init method to be used (xavier / he)
            bias_init (str): The bias init method to be used
            rounding_place (int): The decimal place to round the generated weights and biases
        """
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
        """
        Fits the normalization parameters to the x and y datasets


        Args:
            trainx (array): The input values of the dataset
            trainy (array): The output values of the dataset
            norm_func (function): A normalization function from the equations file
        """
        # check for the norm function passed, then assign the variables needed to execute that norm function
        if norm_func == Normalization.n_zscore: # type: ignore
            self.normdata['mean-x'], self.normdata['stddev-x'] = np.mean(trainx), np.std(trainx)
            self.normdata['mean-y'], self.normdata['stddev-y'] = np.mean(trainy), np.std(trainy)
        elif norm_func == Normalization.n_minmax: # type: ignore
            self.normdata['min-x'], self.normdata['max-x'] = np.amin(trainx), np.amax(trainx)
            self.normdata['min-y'], self.normdata['max-y'] = np.amin(trainy), np.amax(trainy)
           
    def undo_norm(self, y) -> tuple:
        """
        De-normalizes the outputs based on the saved parameters


        Args:
            y: the outputs
        """
        if np.isin(np.array(['mean-x', 'mean-y', 'stddev-x', 'stddev-y']), list(self.normdata.keys())).all():
            return Normalization.n_zscore(y, self.normdata['mean-y'], self.normdata['stddev-y'], inverse=True) # type: ignore
        elif np.isin(np.array(['min-x', 'min-y', 'max-x', 'max-y']), list(self.normdata.keys())).all():
            return Normalization.n_zscore(y, self.normdata['min-y'], self.normdata['max-y'], inverse=True) # type: ignore
        else:
            return y


    def norm(self, x: np.ndarray = None, y: np.ndarray = None) -> tuple:
        """
        Applies the normalization function to the given data


        Args:
            x (array): The input values of the dataset
            y (array): The output values of the dataset
        """
        if np.isin(np.array(['mean-x', 'mean-y', 'stddev-x', 'stddev-y']), list(self.normdata.keys())).all():
            return None if x is None else Normalization.n_zscore(x, self.normdata['mean-x'], self.normdata['stddev-x']), None if y is None else Normalization.n_zscore(y, self.normdata['mean-y'], self.normdata['stddev-y']) # type: ignore
        elif np.isin(np.array(['min-x', 'min-y', 'max-x', 'max-y']), list(self.normdata.keys())).all():
            return None if x is None else Normalization.n_minmax(x, self.normdata['min-x'], self.normdata['max-x']), None if y is None else Normalization.n_zscore(y, self.normdata['min-y'], self.normdata['max-y']) # type: ignore
        return (x, y)
   
    @classmethod    
    def tt_divide(self, x: np.ndarray, y: np.ndarray, training_portion: float = 0.8, testing_portion: float = 0.2) -> tuple:
        """
        Divides the data into stochastic training vs. testing datasets


        Args:
            x (array): The input values of the dataset
            y (array): The output values of the dataset
            training_portion (float): the training side of the ratio between training and testing
            testing_portion (float): the testing side of the ratio between training and testing
        """
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
        """
        Divides the data into mini-batches for training


        Args:
            trainx (array): The input values of the dataset
            trainy (array): The output values of the dataset
            batch_size (int): The sizes of the training mini-batches
        """
        batch_count = int(np.floor(len(trainx) / batch_size))
        xbatches = np.array([trainx[i*batch_size:(i+1)*batch_size] for i in range(batch_count)])
        ybatches = np.array([trainy[i*batch_size:(i+1)*batch_size] for i in range(batch_count)])
        return xbatches, ybatches
   
    def prepdata(self, x: np.ndarray, y: np.ndarray, training_portion: float = 0.8, testing_portion: float = 0.2, batch_size: int = 1, norm_func = None):
        """
        Prepares all of the data for training


        Args:
            x (array): The input values of the dataset
            y (array): The output values of the dataset
            batch_size (int): The sizes of the training mini-batches
            training_portion (float): the training side of the ratio between training and testing
            testing_portion (float): the testing side of the ratio between training and testing
            batch_size (int): The sizes of the training mini-batches
            norm_func (function): A normalization function from the equations file
        """
        trainx, trainy, testx, testy = MultilayerPerceptron.tt_divide(x, y, training_portion, testing_portion)
        if norm_func is not None:
            self.fit_norm(trainx, trainy, norm_func)
            trainx = self.norm(x=trainx)[0]
            testx = self.norm(x=testx)[0]
        trainx, trainy = MultilayerPerceptron.batch_div(trainx, trainy, batch_size)
        return trainx, trainy, testx, testy


# ------------------------
# SAVING
# ------------------------



    def save(self, path: str) -> None:
        """
        Saves a network parameters to a certain path


        Args:
            net (MultilayerPerceptron obj): The network to save
            path (str): The path to save
        """
        with open(path, 'wb') as file:
            pickle.dump(self, file)
    
    @classmethod
    def load(cls, path: str) -> object:
        """
        Loads a network from a specified path (returns a MultilayerPerceptron obj)


        Args:
            path (str): The path to load
        """
        with open(path, 'rb') as file:
            return pickle.load(file)


# ------------------------
# PROCESSING / PROPAGATION
# ------------------------
   
    # RUN A GIVEN VALUE THROUGH THE NETWORK
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Uses a network to predict the outputs for a set of inputs


        Args:
            x (array): The set of inputs to feed through the network
        """
        layer_vals = np.array(x)
        # iterate through the layers via weight matrices for ease
        for i, layer in enumerate(self.layers[1:]):
            layer_vals = layer.activation(np.dot(np.insert(layer_vals, 0, 1, axis=-1), np.insert(self.weight_matrices[i], 0, self.bias_vectors[i], axis=0)))
           
        return layer_vals
   
   


    def train(self, trainx: np.ndarray, trainy: np.ndarray, testx: np.ndarray, testy: np.ndarray, batch_size: int, optimizer: object, epochs: int = 100, metrics: list = ['loss']) -> None:
        """
        Trains the network based on a dataset, uses STD


        Args:
            trainx (array): The set of inputs to feed through the network for training
            trainy (array): The set of outputs to feed through the network for training
            testx (array): The set of inputs to feed through the network for testing
            testy (array): The set of outputs to feed through the network for testing
            batch_size (int): Size of batches
            epochs (int): Number of training cycles
            optimizer (object): Optimizer object (i.e. adam, momentum).
            metrics (list): Metrics to display after epochs and save for graphing
           
        """
       
        if len(trainx) != len(trainy):
            raise ValueError(f'Input and output data are unequal ({len(trainx)} : {len(trainy)})')
        
        # Reshape trainx and trainy to discard batches for use when calculating loss
        rtrainx = trainx.reshape(-1, trainx.shape[-1])
        rtrainy = trainy.reshape(-1, trainy.shape[-1])
        
        # Pre-training metric check
        self.metric_data = []
        train_met = [self.eval_metrics(self.predict(rtrainx), rtrainy, metrics=metrics)]
        test_met = [self.eval_metrics(self.predict(testx), testy, metrics=metrics)]
       
        print(f'Pre-Training Metrics: {test_met[0]}')
       
        for epoch in range(epochs):
            for x, y in zip(trainx, trainy):
               
                # ========================
                # FORWARD PASS
                # ========================
               
                # for use during backwards pass
                weighted_sums = [x]
                activated_sums = [x]
               
                # iterate through the layers
                for i, layer in enumerate(self.layers[1:]):
                    # calc the weighted sum en masse (by batch)
                    z = np.dot(np.insert(activated_sums[i], 0, 1, axis=-1), np.insert(self.weight_matrices[i], 0, self.bias_vectors[i], axis=0))
                    weighted_sums.append(z)
                    activated_sums.append(layer.activation(z))
               
                # ========================
                # BACKWARD PASS
                # ========================
               
                # Use the passed optimizer to complete the backward pass
                weight_updates, bias_updates = optimizer.backpass(x=x, y=y, network=self, batch_size=batch_size, weighted_sums=weighted_sums, activated_sums=activated_sums)
               
                # Unpack all weight / bias updates
                for i, (weight_matrix, bias_vector) in enumerate(zip(weight_updates, bias_updates)):
                    self.weight_matrices[i] -= weight_matrix
                    self.bias_vectors[i] -= bias_vector
                   
            # Save training and testing metric data for graphing
            train_met.append(self.eval_metrics(self.predict(rtrainx), rtrainy, metrics=metrics))
            test_met.append(self.eval_metrics(self.predict(testx), testy, metrics=metrics))
           
            # Display the metrics at the end of each epoch
            if not np.isfinite(np.mean(self.loss_func(self.predict(testx), testy))):
                raise RuntimeError(f'Parameter gradients overflowed ({np.mean(self.loss_func(self.predict(testx), testy))}).')
            print(f'Epoch {epoch+1} / {epochs} Completed. Test metrics: {test_met[-1]}')

        # Reshape and save cumulative epoch metric data
        self.metric_data = {metric: ([epoch_metrics[metric] for epoch_metrics in train_met], [epoch_metrics[metric] for epoch_metrics in test_met]) for metric in metrics}



# ------------------------
# POST-PROCESSING
# ------------------------  
     
     
       
    def graph_metrics(self, epochs: int) -> None:
        """
        Graphs training / test metrics collected from backpropagation


        Args:
            input_dtp (array): The inputs for the network
            input_dtp (array): The inputs for the network
        """
        for metric, (train_data, test_data) in self.metric_data.items():
            plt.figure()
            plt.plot(range(0, epochs+1), train_data, color='red', label=f'Training {metric.title()}')
            plt.plot(range(0, epochs+1), test_data, color='blue', label=f'Testing {metric.title()}')
            plt.xlabel('Epoch')
            plt.ylabel(f'{metric.title()}')
            plt.title(f'{metric.title()} by Epoch')
            plt.grid(True)
            plt.legend()
            plt.gcf().canvas.manager.set_window_title(metric)
        plt.show()    


    def eval_metrics(self, x: np.ndarray, y: np.ndarray, metrics = []):
        """
        Evaluates the network based on specified metrics (loss, accuracy, MAE, MSE)


        Args:
            inps (array): The inputs for the network
            outs (array): The inputs for the network
            metrics (list of str): The list of metrics to be used
        """
        def loss(y_pred: np.ndarray, y_actu: np.ndarray):
            return np.mean(self.loss_func(y_pred, y_actu))
       
        def accuracy(y_pred: np.ndarray, y_actu: np.ndarray):
            amx_actu = np.argmax(y_actu, axis=-1)
            return np.mean(np.argmax(y_pred, axis=-1) == amx_actu)
       
        def mae(y_pred: np.ndarray, y_actu: np.ndarray):
            return np.mean(np.abs(y_actu - y_pred))
       
        def mse(y_pred: np.ndarray, y_actu: np.ndarray):
            return np.mean(np.power(y_actu - y_pred, 2))
       
        return {metric: {'loss': loss, 'accuracy': accuracy, 'mae': mae, 'mse': mse}[metric](x, y) for metric in metrics}
   


# ------------------------
# END OF FNN FRAMEWORK
# ------------------------  
