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