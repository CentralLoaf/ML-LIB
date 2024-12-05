import numpy as np

class MiniBatchGD:
    def __init__(self, lr: float = 0.001):
        self.lr = lr
    
    def backpass(self, x: np.ndarray, y: np.ndarray, network: object, batch_size: int, weighted_sums: list, activated_sums: list):
        
        # ========================
        # BACKWARD PASS (OUTPUT LAYER)
        # ======================== 
        
        # The gradient of the node-specific cost (vector by sample, matrix by batch)
        dx_cost = network.loss_func(y_pred=activated_sums[-1], y_actu=y, dx=True)
        
        # The gradient of the activated sum (vector by sample, matrix by batch)
        dx_out_actisum = network.layers[-1].activation(weighted_sums[-1], dx=True)
        
        # error term for the output layer, *inputs for weights, *1 for biases
        eterms = dx_cost * dx_out_actisum
        
        # Gradients for params
        w_grad = np.dot(activated_sums[-2].T, eterms) / batch_size
        b_grad = np.sum(eterms, axis=0) / batch_size
        
        # Parameter updates using param gradients * learning rate
        weight_updates = [-w_grad * self.lr]
        bias_updates = [-b_grad * self.lr]
        
        # ========================
        # BACKWARD PASS (HIDDEN LAYERS)
        # ========================
        
        for i, weights in zip(reversed(list(range(1, len(network.weight_matrices)))), reversed(network.weight_matrices[1:])):
            
            dx_actisum = network.layers[i].activation(weighted_sums[i], dx=True)
            
            # Error term for the hidden layers, *inputs for weights, *1 for biases 2 4 9 1
            eterms = np.dot(eterms, weights.T) * dx_actisum
            
            # Gradients for params
            w_grad = np.dot(activated_sums[i-1].T, eterms) / batch_size
            b_grad = np.sum(eterms, axis=0) / batch_size
            
            # Parameter updates using param gradients * learning rate
            weight_updates.insert(0, -w_grad * self.lr)
            bias_updates.insert(0, -b_grad * self.lr)
            
        return weight_updates, bias_updates
    
# ------------------------
    
class MomentumGD:
    def __init__(self, lr: float = 0.001, momentum_factor: float = 0.9):
        self.lr = lr
        self.momentum_factor = momentum_factor
        
    def init(self, network: object):
        self.velocity_w, self.velocity_b = [], []
        # Unpack by layer, build up zero-arrays from there
        for weight_matrix, bias_vector in zip(network.weight_matrices, network.bias_vectors):
            self.velocity_w.append(np.zeros_like(weight_matrix))
            self.velocity_b.append(np.zeros_like(bias_vector))
    
    def backpass(self, x: np.ndarray, y: np.ndarray, network: object, batch_size: int, weighted_sums: list, activated_sums: list):
        
        # ========================
        # BACKWARD PASS (OUTPUT LAYER)
        # ======================== 
        
        # The gradient of the node-specific cost (vector by sample, matrix by batch)
        dx_cost = network.loss_func(y_pred=activated_sums[-1], y_actu=y, dx=True)
        
        # The gradient of the activated sum (vector by sample, matrix by batch)
        dx_out_actisum = network.layers[-1].activation(weighted_sums[-1], dx=True)
        
        # error term for the output layer, *inputs for weights, *1 for biases
        eterms = dx_cost * dx_out_actisum
        
        # Gradients for params
        w_grad = np.dot(activated_sums[-2].T, eterms) / batch_size
        b_grad = np.sum(eterms, axis=0) / batch_size
        
        # Update velocities
        self.velocity_w[-1] = (self.momentum_factor * self.velocity_w[-1]) - (w_grad * self.lr)
        self.velocity_b[-1] = (self.momentum_factor * self.velocity_b[-1]) - (b_grad * self.lr)
        
        # Parameter updates using param gradients * learning rate
        weight_updates = [self.velocity_w[-1]]
        bias_updates = [self.velocity_b[-1]]
        
        # ========================
        # BACKWARD PASS (HIDDEN LAYERS)
        # ========================
        
        for i, weights in zip(reversed(list(range(1, len(network.weight_matrices)))), reversed(network.weight_matrices[1:])):
            dx_actisum = network.layers[i].activation(weighted_sums[i], dx=True)
            
            # Error term for the hidden layers, *inputs for weights, *1 for biases 2 4 9 1
            eterms = np.dot(eterms, weights.T) * dx_actisum
            
            # Gradients for params
            w_grad = np.dot(activated_sums[i-1].T, eterms) / batch_size
            b_grad = np.sum(eterms, axis=0) / batch_size
            
            # Update velocities
            self.velocity_w[i-1] = (self.momentum_factor * self.velocity_w[i-1]) - (w_grad * self.lr)
            self.velocity_b[i-1] = (self.momentum_factor * self.velocity_b[i-1]) - (b_grad * self.lr)
            
            # Parameter updates using param gradients * learning rate
            weight_updates.insert(0, self.velocity_w[i-1])
            bias_updates.insert(0, self.velocity_b[i-1])
            
        return weight_updates, bias_updates
    
# ------------------------
    
class Adagrad:
    def __init__(self, lr: float = 0.001):
        self.lr = lr
        
    def init(self, network: object):
        self.adalr_w, self.adalr_b = [], []
        # Unpack by layer, build up zero-arrays from there
        for weight_matrix, bias_vector in zip(network.weight_matrices, network.bias_vectors):
            self.adalr_w.append(np.zeros_like(weight_matrix))
            self.adalr_b.append(np.zeros_like(bias_vector))
    
    def backpass(self, x: np.ndarray, y: np.ndarray, network: object, batch_size: int, weighted_sums: list, activated_sums: list):
        
        # ========================
        # BACKWARD PASS (OUTPUT LAYER)
        # ======================== 
        
        # The gradient of the node-specific cost (vector by sample, matrix by batch)
        dx_cost = network.loss_func(y_pred=activated_sums[-1], y_actu=y, dx=True)
        
        # The gradient of the activated sum (vector by sample, matrix by batch)
        dx_out_actisum = network.layers[-1].activation(weighted_sums[-1], dx=True)
        
        # error term for the output layer, *inputs for weights, *1 for biases
        eterms = dx_cost * dx_out_actisum
        
        # Gradients for params
        w_grad = np.dot(activated_sums[-2].T, eterms) / batch_size
        b_grad = np.sum(eterms, axis=0) / batch_size
        
        # Update velocities
        self.adalr_w[-1] += np.square(w_grad)
        self.adalr_b[-1] += np.square(b_grad)
        
        # Parameter updates using param gradients * learning rate
        weight_updates = [-(self.lr / np.sqrt(self.adalr_w[-1] + 1e-10)) * w_grad]
        bias_updates = [-(self.lr / np.sqrt(self.adalr_b[-1] + 1e-10)) * b_grad]
        
        # ========================
        # BACKWARD PASS (HIDDEN LAYERS)
        # ========================
        
        for i, weights in zip(reversed(list(range(1, len(network.weight_matrices)))), reversed(network.weight_matrices[1:])):
            dx_actisum = network.layers[i].activation(weighted_sums[i], dx=True)
            
            # Error term for the hidden layers, *inputs for weights, *1 for biases 2 4 9 1
            eterms = np.dot(eterms, weights.T) * dx_actisum
            
            # Gradients for params
            w_grad = np.dot(activated_sums[i-1].T, eterms) / batch_size
            b_grad = np.sum(eterms, axis=0) / batch_size
            
            # Update velocities
            self.adalr_w[i-1] += np.square(w_grad)
            self.adalr_b[i-1] += np.square(b_grad)
            
            # Parameter updates using param gradients * learning rate
            weight_updates.insert(0, -(self.lr / np.sqrt(self.adalr_w[i-1] + 1e-10)) * w_grad)
            bias_updates.insert(0, -(self.lr / np.sqrt(self.adalr_b[i-1] + 1e-10)) * b_grad)
            
        return weight_updates, bias_updates
    
# ------------------------
    
class RMSprop(Adagrad):
    def __init__(self, lr: float = 0.001, decay_rate: float = 0.9):
        self.lr = lr
        self.decay = decay_rate
    
    def backpass(self, x: np.ndarray, y: np.ndarray, network: object, batch_size: int, weighted_sums: list, activated_sums: list):
        
        # ========================
        # BACKWARD PASS (OUTPUT LAYER)
        # ======================== 
        
        # The gradient of the node-specific cost (vector by sample, matrix by batch)
        dx_cost = network.loss_func(y_pred=activated_sums[-1], y_actu=y, dx=True)
        
        # The gradient of the activated sum (vector by sample, matrix by batch)
        dx_out_actisum = network.layers[-1].activation(weighted_sums[-1], dx=True)
        
        # error term for the output layer, *inputs for weights, *1 for biases
        eterms = dx_cost * dx_out_actisum
        
        # Gradients for params
        w_grad = np.dot(activated_sums[-2].T, eterms) / batch_size
        b_grad = np.sum(eterms, axis=0) / batch_size
        
        # Update velocities
        self.adalr_w[-1] += (1 - self.decay) * np.square(w_grad)
        self.adalr_b[-1] += (1 - self.decay) * np.square(b_grad)
        
        # Parameter updates using param gradients * learning rate
        weight_updates = [-(self.lr / np.sqrt(self.adalr_w[-1] + 1e-10)) * w_grad]
        bias_updates = [-(self.lr / np.sqrt(self.adalr_b[-1] + 1e-10)) * b_grad]
        
        # ========================
        # BACKWARD PASS (HIDDEN LAYERS)
        # ========================
        
        for i, weights in zip(reversed(list(range(1, len(network.weight_matrices)))), reversed(network.weight_matrices[1:])):
            dx_actisum = network.layers[i].activation(weighted_sums[i], dx=True)
            
            # Error term for the hidden layers, *inputs for weights, *1 for biases 2 4 9 1
            eterms = np.dot(eterms, weights.T) * dx_actisum
            
            # Gradients for params
            w_grad = np.dot(activated_sums[i-1].T, eterms) / batch_size
            b_grad = np.sum(eterms, axis=0) / batch_size
            
            # Update velocities
            self.velocity_w[i-1] = (self.momentum * self.velocity_w[i-1]) - (w_grad * self.lr)
            self.velocity_b[i-1] = (self.momentum * self.velocity_b[i-1]) - (b_grad * self.lr)
            
            # Update velocities
            self.adalr_w[i-1] += np.square(w_grad)
            self.adalr_b[i-1] += np.square(b_grad)
            
            # Parameter updates using param gradients * learning rate
            weight_updates.insert(0, -(self.lr / np.sqrt(self.adalr_w[i-1] + 1e-10)) * w_grad)
            bias_updates.insert(0, -(self.lr / np.sqrt(self.adalr_b[i-1] + 1e-10)) * b_grad)
            
        return weight_updates, bias_updates
    
# ------------------------

class Adam:
    def __init__(self, lr: float = 0.001, decay_rate1: float = 0.9, decay_rate2: float = 0.999):
        self.lr = lr
        self.decay_rate1 = decay_rate1
        self.decay_rate2 = decay_rate2
        
    def init(self, network: object):
        self.momentum_w, self.momentum_b = [], []
        self.velocity_w, self.velocity_b = [], []
        self.timestep = 0
        
        # Unpack by layer, build up zero-arrays from there
        for weight_matrix, bias_vector in zip(network.weight_matrices, network.bias_vectors):
            # Step #1, MomentumGD
            self.momentum_w.append(np.zeros_like(weight_matrix))
            self.momentum_b.append(np.zeros_like(bias_vector))
            # Step #2, RMSprop
            self.velocity_w.append(np.zeros_like(weight_matrix))
            self.velocity_b.append(np.zeros_like(bias_vector))
    
    def backpass(self, x: np.ndarray, y: np.ndarray, network: object, batch_size: int, weighted_sums: list, activated_sums: list):
        
        # ========================
        # BACKWARD PASS (OUTPUT LAYER)
        # ======================== 
        
        # Incrementing the timestep for bias correction
        self.timestep += 1
        
        # The gradient of the node-specific cost (vector by sample, matrix by batch)
        dx_cost = network.loss_func(y_pred=activated_sums[-1], y_actu=y, dx=True)
        
        # The gradient of the activated sum (vector by sample, matrix by batch)
        dx_out_actisum = network.layers[-1].activation(weighted_sums[-1], dx=True)
        
        # error term for the output layer, *inputs for weights, *1 for biases
        eterms = dx_cost * dx_out_actisum
        
        # Gradients for params
        w_grad = np.dot(activated_sums[-2].T, eterms) / batch_size
        b_grad = np.sum(eterms, axis=0) / batch_size
        
        # Step #1- Apply MomentumGD learning
        self.momentum_w[-1] = (self.decay_rate1 * self.momentum_w[-1]) + (1 - self.decay_rate1) * w_grad
        self.momentum_b[-1] = (self.decay_rate1 * self.momentum_b[-1]) + (1 - self.decay_rate1) * b_grad
        
        # Step #2- Apply RMSprop learning
        self.velocity_w[-1] = (self.decay_rate2 * self.velocity_b[-1]) + (1 - self.decay_rate2) * np.square(w_grad)
        self.velocity_b[-1] = (self.decay_rate2 * self.velocity_b[-1]) + (1 - self.decay_rate2) * np.square(b_grad)
        
        # Step #3- Bias correction
        corrected_momentum_w = self.momentum_w[-1] / (1 - np.pow(self.decay_rate1, self.timestep))
        corrected_momentum_b = self.momentum_b[-1] / (1 - np.pow(self.decay_rate1, self.timestep))
        
        corrected_velocity_w = self.velocity_w[-1] / (1 - np.pow(self.decay_rate2, self.timestep))
        corrected_velocity_b = self.velocity_b[-1] / (1 - np.pow(self.decay_rate2, self.timestep))
        
        # Parameter updates using param gradients * learning rate
        weight_updates = [-(self.lr / (np.sqrt(corrected_velocity_w) + 1e-10)) * corrected_momentum_w]
        bias_updates = [-(self.lr / (np.sqrt(corrected_velocity_b) + 1e-10)) * corrected_momentum_b]
        
        # ========================
        # BACKWARD PASS (HIDDEN LAYERS)
        # ========================
        
        for i, weights in zip(reversed(list(range(1, len(network.weight_matrices)))), reversed(network.weight_matrices[1:])):
            dx_actisum = network.layers[i].activation(weighted_sums[i], dx=True)
            
            # Error term for the hidden layers, *inputs for weights, *1 for biases 2 4 9 1
            eterms = np.dot(eterms, weights.T) * dx_actisum
            
            # Gradients for params
            w_grad = np.dot(activated_sums[i-1].T, eterms) / batch_size
            b_grad = np.sum(eterms, axis=0) / batch_size
            
            # Step #1- Apply MomentumGD learning
            self.momentum_w[i-1] = (self.decay_rate1 * self.momentum_w[i-1]) + (1 - self.decay_rate1) * w_grad
            self.momentum_b[i-1] = (self.decay_rate1 * self.momentum_b[i-1]) + (1 - self.decay_rate1) * b_grad
            
            # Step #2- Apply RMSprop learning
            self.velocity_w[i-1] = (self.decay_rate2 * self.velocity_w[i-1]) + (1 - self.decay_rate2) * np.square(w_grad)
            self.velocity_b[i-1] = (self.decay_rate2 * self.velocity_b[i-1]) + (1 - self.decay_rate2) * np.square(b_grad)
            
            # Step #3- Bias correction
            corrected_momentum_w = self.momentum_w[i-1] / (1 - np.pow(self.decay_rate1, self.timestep))
            corrected_momentum_b = self.momentum_b[i-1] / (1 - np.pow(self.decay_rate1, self.timestep))
            
            corrected_velocity_w = self.velocity_w[i-1] / (1 - np.pow(self.decay_rate2, self.timestep))
            corrected_velocity_b = self.velocity_b[i-1] / (1 - np.pow(self.decay_rate2, self.timestep))
            
            # Parameter updates using param gradients * learning rate
            weight_updates.insert(0, -(self.lr / (np.sqrt(corrected_velocity_w) + 1e-10)) * corrected_momentum_w)
            bias_updates.insert(0, -(self.lr / (np.sqrt(corrected_velocity_b) + 1e-10)) * corrected_momentum_b)
            
        return weight_updates, bias_updates
    
# ------------------------