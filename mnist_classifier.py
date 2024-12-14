from mlib.algorithms.mlp import *
from mlib.equations import *
from mlib.optimizers import *
import numpy as np


import tensorflow as tf
                 
net = MultilayerPerceptron(Layer(784), Layer(360, Activations.relu), Layer(120, Activations.relu), Layer(40, Activations.relu), Layer(10, Activations.sigmoid), loss_func=Loss.bce)
net.generate_params(weight_init='xaviar')


# FORMATTING AND LOADING MNIST
mnist = tf.keras.datasets.mnist
(train_pix, train_lab), (test_pix, test_lab) = mnist.load_data()


pix = np.concatenate((train_pix / 255, test_pix / 255))
pix = pix.reshape(pix.shape[0], -1)
lab = np.concatenate((np.array([np.array([1 if i == digit else 0 for i in range(10)]) for digit in train_lab]), np.array([np.array([1 if i == digit else 0 for i in range(10)]) for digit in test_lab])))
#lab = np.concatenate((np.array([np.array([1 if digit >= 5 else 0]) for digit in train_lab]), np.array([np.array([1 if digit >= 5 else 0]) for digit in test_lab])))
#lab = np.concatenate((train_lab, test_lab))


# TRAINING AND DATA
train_pix, train_lab, test_pix, test_lab = net.prepdata(pix, lab, 0.8, 0.2, 64, None)


'''op = Adam(lr=0.0001)
op.init(net)
net.train(train_pix, train_lab, test_pix, test_lab, batch_size=64, optimizer=op, epochs=20, metrics=['loss', 'accuracy'])
net.save('mnist_classifier.pkl')
net.graph_metrics(20)'''

net = MultilayerPerceptron.load("mnist_classifier.pkl")

for digit, pred, prob in [(np.argmax(test_lab[i]), np.argmax(net.predict(test_pix[i])), net.predict(test_pix[i])) for i in range(100)]:
        print(f'DIGIT: {digit}, PRED: {pred}, PROB: {prob}')
        

'''x = np.array([[15, 10], [15, 10], [15, 10], [15, 10], [15, 10], [15, 20], [15, 20], [15, 20], [15, 20], [15, 20], [15, 30], [15, 30], [15, 30], [15, 30], [15, 30], [15, 40], [15, 40], [15, 40], [15, 40], [15, 40], [15, 50], [15, 50], [15, 50], [15, 50], [15, 50],
          [30, 10], [30, 10], [30, 10], [30, 10], [30, 10], [30, 20], [30, 20], [30, 20], [30, 20], [30, 20], [30, 30], [30, 30], [30, 30], [30, 30], [30, 30], [30, 40], [30, 40], [30, 40], [30, 40], [30, 40], [30, 50], [30, 50], [30, 50], [30, 50], [30, 50],
          [45, 10], [45, 10], [45, 10], [45, 10], [45, 10], [45, 20], [45, 20], [45, 20], [45, 20], [45, 20], [45, 30], [45, 30], [45, 30], [45, 30], [45, 30], [45, 40], [45, 40], [45, 40], [45, 40], [45, 40], [45, 50], [45, 50], [45, 50], [45, 50], [45, 50],
          [60, 10], [60, 10], [60, 10], [60, 10], [60, 10], [60, 20], [60, 20], [60, 20], [60, 20], [60, 20], [60, 30], [60, 30], [60, 30], [60, 30], [60, 30], [60, 40], [60, 40], [60, 40], [60, 40], [60, 40], [60, 50], [60, 50], [60, 50], [60, 50], [60, 50]])
y = np.array([[5.68], [5], [5.88], [4.72], [5.28], [6.03], [6.06], [5.69], [6], [5.78], [7.35], [7.43], [7.32], [7.25], [6.79], [8.63], [8.78], [8.82], [8], [8.84], [10.03], [10.69], [10.9], [9.65], [11.69],
        [5.63], [5.54], [5.31], [5.41], [5.28], [5.88], [6.53], [6.46], [6.01], [6.22], [7.1], [7.19], [6.92], [6.97], [7.16], [8.13], [8.93], [8.97], [8.34], [9.25], [11.22], [11.16], [10.5], [11.28], [11.12],
        [5.41], [5.53], [5.69], [5.65], [5.53], [6.4], [6.38], [5.78], [5.88], [6.93], [8.03], [7.59], [7.6], [8.06], [8.1], [9.53], [9.69], [9.69], [9.94], [10.16], [12.94], [12.28], [12.88], [13], [12.77],
        [6.09], [6.03], [5.53], [6.18], [5.56], [7.28], [7], [7.18], [6.56], [6.75], [7.81], [8.31], [8.65], [8.84], [8.78], [11.16], [10.06], [10.63], [10.94], [11.15], [13.06], [13.07], [13.87], [13.5], [14.25]])


met = MultilayerPerceptron(Layer(2),
                              Layer(4, Activations.elu),
                              Layer(9, Activations.elu),
                              Layer(1, Activations.linear), loss_func=Loss.mse)
met.generate_params()


trainx, trainy, testx, testy = met.prepdata(x, y, batch_size=8)

optimizer = Adam(lr=0.001)
optimizer.init(met)
met.train(trainx, trainy, testx, testy, batch_size=8, epochs=10, optimizer=optimizer, metrics=['loss', 'mae'])
met.graph_metrics(10)'''


'''inps = np.array([np.array([x]) for x in np.array(range(500)) / 500])
outs = np.array([np.array([3*x+1]) for x in np.array(range(500)) / 500])


met = mp(Layer(1), Layer(1, Activations.linear), loss_func=Loss.mse)
met.generate_params(weight_init='xaviar', bias_init='small-normal')


trainx, trainy, testx, testy = met.prepdata(inps, outs, 0.8, 0.2, 8, None)


print(met.eval_metrics(inps, outs, ['mae']))


op = Adam(lr=0.01)
op.init(met)
met.train(trainx, trainy, testx, testy, batch_size=32, epochs=400, optimizer=op)
print(met.eval_metrics(inps, outs, ['mae']), met.weight_matrices, met.bias_vectors)'''






'''net = MultilayerPerceptron(Layer(2),
                              Layer(3, Activations.elu),
                              Layer(1, Activations.elu), loss_func=Loss.mse)


net.weight_matrices = [np.array([np.array([0.1, 0.6, -0.4]), np.array([-0.5, 0.2, 1.0])]), np.array([np.array([-0.8]), np.array([0.2]), np.array([0.3])])]
net.bias_vectors = [np.array([1.0, 2.5, -1.5]), np.array([1.5])]


print(net)


trainx, trainy, testx, testy = net.prepdata([np.array([2, 3]), np.array([2, 3])], [np.array([0]), np.array([0])], 0.5, 0.5, 1, None)
net.train(trainx, trainy, testx, testy, epochs=1, lr=0.0001, batch_size=1)'''

