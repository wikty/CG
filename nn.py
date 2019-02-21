import random
import math

import matplotlib.pyplot as plt

from cg import Variable, F


class MSELoss(object):

    def fn(self, outputs, targets):
        """Return the MSE loss of a mini-batch."""
        if (not outputs) and (not outputs):
            return 0.0
        
        assert len(outputs) == len(targets)
        assert len(outputs[0]) == len(targets[0])
        
        loss = 0.0
        for output, target in zip(outputs, targets):
            for o_d, t_d in zip(output, target):
                loss += (o_d - t_d) ** 2
        return loss / len(outputs)

    def grad(self, outputs, targets):
        assert outputs and targets
        assert len(outputs) == len(targets)
        for i in range(len(outputs)):
            assert len(outputs[i]) == len(targets[i])

        grads = [0.0 for i in range(len(outputs[0]))]
        for n, output, target in zip(range(len(outputs)), outputs, targets):
            for i, o, t in zip(range(len(output)), output, target):
                # mean
                grads[i] = grads[i]*(n/(n+1)) + (2*(o - t)) / (n+1)
        return grads


class Neuron(object):

    def __init__(self, in_dim):
        self.in_dim = in_dim
        self.bias = Variable()
        self.weights = [Variable() for i in range(self.in_dim)]
        
    def init_params(self):
        """Init parameters with standard gaussian distribution."""
        for w in self.weights:
            w.set_value(random.gauss(0, 1), False)
        self.bias.set_value(random.gauss(0, 1), False)

    def params(self):
        return self.weights + [self.bias]

    def forward(self, inputs):
        assert len(inputs) == self.in_dim
        output = sum([v*w for v, w in zip(inputs, self.weights)])
        return F.relu(output + self.bias)


class Layer(object):

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.neurons = [
            Neuron(self.in_dim) for i in range(self.out_dim)
        ]

    def init_params(self):
        for neuron in self.neurons:
            neuron.init_params()

    def params(self):
        l = []
        for neuron in self.neurons:
            l.extend(neuron.params())
        return l

    def forward(self, inputs):
        assert len(inputs) == self.in_dim
        return [
            neuron.forward(inputs) for neuron in self.neurons
        ]


class Network(object):

    def __init__(self, sizes):
        """`sizes` is a list of the size of each layer."""
        self.sizes = sizes
        self.layers = [
            Layer(in_dim, out_dim) 
            for in_dim, out_dim in zip(sizes[:-1], sizes[1:])
        ]
        self.outputs = []
        self.init_params()

    def init_params(self):
        for layer in self.layers:
            layer.init_params()
        return self

    def params(self):
        l = []
        for layer in self.layers:
            l.extend(layer.params())
        return l

    def forward(self, inputs):
        """`inputs` is a list of the values of a sample."""
        assert len(inputs) == self.sizes[0]
        inputs = [Variable(value) for value in inputs]
        for layer in self.layers:
            inputs = layer.forward(inputs)
        self.outputs = inputs
        for output in self.outputs:
            output.forward()
        return self

    def backward(self, grads):
        assert len(grads) == self.sizes[-1]
        for output, grad in zip(self.outputs, grads):
            output.backward(grad)
        return self

    def zero_grad(self):
        for output in self.outputs:
            output.zero_grad(backprop=True)
        return self

    def predict(self, inputs):
        self.forward(inputs)

        return [o.value for o in self.outputs]

    def dump_graph(self):
        for output in self.outputs:
            output.dump_graph()
        return self

    def SGD(self, mini_batch, loss_fn, lr=0.001):
        self.zero_grad()
        batch_size = len(mini_batch)
        outputs = []
        targets = []
        for x, y in mini_batch:
            targets.append(y)
            # forward
            outputs.append(self.predict(x))
        grads = loss_fn.grad(outputs, targets)
        # backward
        self.backward(grads)
        # update params
        for param in self.params():
            param.set_value(-lr * param.grad, True)
        return self


def load_data(fns, args, in_dim, out_dim, n=100,
    minval=-10, maxval=10):
    def sigmoid(z):
        try:
            return 1 / (1 + math.exp(-z))
        except OverflowError:
            return 0.0
    fns = fns * (int(in_dim/len(fns))+1)
    args = args * (int(in_dim/len(fns))+1)
    x_list, y_list = [], []
    for i in range(n):
        x = []
        y = [None for k in range(out_dim)]
        for j in range(in_dim):
            x.append(random.randint(0, maxval-minval) + minval)
        x_list.append(x)
        y_list.append(y)
    for j in range(out_dim):
        random.shuffle(fns)
        random.shuffle(args)
        for i in range(n):
            y = 0.0
            for x, fn, arg in zip(x_list[i], fns, args):
                y += arg * fn(x)
            y_list[i][j] = sigmoid(y)
    return x_list, y_list


def get_batchs(x_list, y_list, batch_size):
    assert len(x_list) == len(y_list)
    mini_batches = []
    n = len(x_list)
    for i in range(0, n, batch_size):
        batch_x = x_list[i:i+batch_size]
        batch_y = y_list[i:i+batch_size]
        mini_batches.append([
            (x, y) for x, y in zip(batch_x, batch_y)
        ])
    random.shuffle(mini_batches)
    return mini_batches


def evaluate(net, loss_fn, x, y):
    outputs = []
    targets = []
    for i in range(len(x)):
        targets.append(y[i])
        outputs.append(net.predict(x[i]))
    return loss_fn.fn(outputs, targets)


if __name__ == '__main__':
    args = [-3.23, 5.8, 0.98]
    fns = [math.sin, math.cos, math.exp]
    in_dim, out_dim = 5, 2
    train_n, test_n = 1000, 100
    x, y = load_data(fns, args, in_dim, out_dim, train_n+test_n)
    train_x, test_x = x[:train_n], x[train_n:]
    train_y, test_y = y[:train_n], y[train_n:]

    loss_fn = MSELoss()
    net = Network(sizes=(in_dim, 10, out_dim))

    epoch = 20
    batch_size = 10
    learning_rate = 0.001
    train_losses = []
    test_losses = []
    for e in range(epoch):
        mini_batches = get_batchs(train_x, train_y, batch_size)
        for i, mini_batch in enumerate(mini_batches):            
            net.SGD(mini_batch, loss_fn, learning_rate)

            test_loss = evaluate(net, loss_fn, test_x, test_y)
            test_losses.append(test_loss)
            print('Epoch/Batch: {}/{}, Loss: {}'.format(e, i, test_loss))
            train_loss = evaluate(net, loss_fn, train_x, train_y)
            train_losses.append(train_loss)

    # plot train&test loss
    plt.plot(train_losses, c='red')
    plt.plot(test_losses, c='green')
    plt.show()