import math

from cg import Node, AddGate, MultiplyGate, SigmoidGate


###
# example: use node and gate to build computational graph.
###
class Neuron(object):
    """f = sigmoid(ax + by + c)

    If you want, you can assume (x, y) is datapoint(fixed), (a, b, c) are 
    parameters(tuned). But there isn't difference between them for now.
    """

    def __init__(self, a, b, c, x, y):
        self.a = Node(a)
        self.b = Node(b)
        self.c = Node(c)
        self.x = Node(x)
        self.y = Node(y)
        h1 = MultiplyGate(self.a, self.x).output()
        h2 = MultiplyGate(self.b, self.y).output()
        h3 = AddGate(h1, h2).output()
        h4 = AddGate(h3, self.c).output()
        self.output = SigmoidGate(h4).output()

    def math(self, a, b, c, x, y):
        n = a * x + b * y + c
        return 1 / (1 + math.exp(-n))

    def math_grad(self, a, b, c, x, y, h=0.001):
        t = self.math(a, b, c, x, y)
        a_grad = (self.math(a+h, b, c, x, y) - t) / h
        b_grad = (self.math(a, b+h, c, x, y) - t) / h
        c_grad = (self.math(a, b, c+h, x, y) - t) / h
        x_grad = (self.math(a, b, c, x+h, y) - t) / h
        y_grad = (self.math(a, b, c, x, y+h) - t) / h
        return a_grad, b_grad, c_grad, x_grad, y_grad

    def grad(self):
        return (self.a.grad, self.b.grad, self.c.grad, 
            self.x.grad, self.y.grad)

    def forward(self):
        return self.output.forward()

    def backward(self, grad=1.0):
        self.output.backward(grad=grad)

    def zero_grad(self):
        self.output.zero_grad(backprop=True)

    def grad_descent(self, lr=0.01):
        self.a.set_value(-lr * self.a.grad, True)
        self.b.set_value(-lr * self.b.grad, True)
        self.c.set_value(-lr * self.c.grad, True)
        self.x.set_value(-lr * self.x.grad, True)
        self.y.set_value(-lr * self.y.grad, True)