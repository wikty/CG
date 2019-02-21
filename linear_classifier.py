import random

import numpy as np
from matplotlib import pyplot as plt

from cg import Variable, F

class LinearClassifier(object):
    """boundary: x1*w1 + x2*w2 + b = 0"""

    def __init__(self):
        # init parameters
        self.w1 = Variable(random.gauss(0, 1))
        self.w2 = Variable(random.gauss(0, 1))
        self.b = Variable(random.gauss(0, 1))
        # input is empty for now
        self.x1 = Variable()
        self.x2 = Variable()
        # define graph
        self.output = (self.x1 * self.w1) + (self.x2 * self.w2) + self.b

    def forward(self, x1_value, x2_value):
        """Return the score of classifier output."""
        self.x1.set_value(x1_value, False)
        self.x2.set_value(x2_value, False)
        self.output.forward()
        return self.output.value

    def backward(self, grad=1.0):
        """Backward propagation gradients."""
        self.output.backward(grad)

    def zero_grad(self):
        self.output.zero_grad(backprop=True)

    def predict(self, x1_value, x2_value):
        """Return the label of classifier output."""
        score = self.forward(x1_value, x2_value)
        if score > 0.0:
            return 1.0
        else:
            return -1.0

    def grad_descent(self, batch_size=1, lr=0.01):
        self.w1.set_value(-lr * (self.w1.grad/batch_size), True)
        self.w2.set_value(-lr * (self.w2.grad/batch_size), True)
        self.b.set_value(-lr * (self.b.grad/batch_size), True)

    def SGD(self, mini_batch, lr=0.01):
        self.zero_grad()
        for (x1, x2), label in mini_batch:
            prediction = self.predict(x1, x2)
            # misclassified loss: -(prediction*label)
            if prediction*label < 0.0:
                grad = -label
                self.backward(grad)
        self.grad_descent(batch_size=len(mini_batch), lr=lr)

    def dump(self):
        return {
            'w1': self.w1.value,
            'w2': self.w2.value,
            'b': self.b.value
        }


def load_data(n=50, w1=None, w2=None, b=None, 
    minval=-100, maxval=100):
    def f(x1, x2):
        noise = random.gauss(0, 1)  # gaussian nosie
        return x1*w1 + x2*w2 + b + noise
    for i in range(n):
        x1 = random.randint(0, maxval-minval) + minval
        x2 = random.randint(0, maxval-minval) + minval

        if f(x1, x2) > 0.0:
            label = 1.0
        else:
            label = -1.0
        yield ((x1, x2), label)


def show_data(dataset):
    x_pos, x_neg = [], []
    for (x1, x2), label in dataset:
        if label < 0.0:
            x_neg.append([x1, x2])
        else:
            x_pos.append([x1, x2])
    x_pos, x_neg = np.array(x_pos), np.array(x_neg)
    plt.scatter(x_pos[:, 0], x_pos[:, 1], c='red', marker='+')
    plt.scatter(x_neg[:, 0], x_neg[:, 1], c='green', marker='o')    


def show_boundary(w1, w2, b, minval, maxval, **kwargs):
    x1 = np.linspace(minval, maxval)
    x2 = -(w1/w2)*x1 + b
    plt.plot(x1, x2, **kwargs)


if __name__ == '__main__':
    print('# Linear Classifier #')
    w1, w2, b = 3.23, -2.18, 5.8
    n, batch_size = 100, 4
    minval, maxval = -100, 100
    print('Function: x1*{} + x2*{} + {} + noise'.format(w1, w2, b))
    train_data_loader = load_data(
        n=100, w1=w1, w2=w2, b=b, minval=minval, maxval=maxval)
    test_data_loader = load_data(
        n=int(n/3), w1=w1, w2=w2, b=b, minval=minval, maxval=maxval)
    train_data = list(train_data_loader)
    test_data = list(test_data_loader) 

    classifier = LinearClassifier()
    for epoch in range(50):
        print('Epoch: {}'.format(epoch))
        # for (x1, x2), label in train_data:
        #     print('Label: {}'.format(label))
        #     prediction = classifier.predict(x1, x2)
        #     print('Before Train Prediction: {}'.format(
        #         prediction
        #     ))
        #     classifier.SGD([((x1, x2), label)])
        #     prediction = classifier.predict(x1, x2)
        #     print('After Train Prediction: {}'.format(
        #         prediction
        #     ))
        random.shuffle(train_data)
        for i in range(0, len(train_data), batch_size):
            classifier.SGD(train_data[i:i+batch_size])
        # evaluate accuary
        corrects = 0
        total = 0
        for (x1, x2), label in test_data:
            total += 1
            prediction = classifier.predict(x1, x2)
            corrects += 1 if prediction == label else 0
        acc = corrects / total
        print('Epoch: {}, Acc: {}'.format(epoch, acc))
    params = classifier.dump()
    print('Classifier Parameters: {}'.format(params))

    print('Show test dataset and boundary')
    fig = plt.figure()
    show_data(test_data)
    # true boundary
    show_boundary(w1, w2, b, minval, maxval, 
        c='blue', label='original boundary')
    show_boundary(params['w1'], params['w2'], params['b'], minval, maxval,
        c='yellow', label='prediction boundary')
    plt.legend()
    plt.show()