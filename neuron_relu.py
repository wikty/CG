import math

from cg import Variable, F


###
# example: use variable to build computational graph
###
class Neuron(object):
    """output = relu(a*x + b*y +c)"""

    def __init__(self, a, b, c, x, y):
        self.a = Variable(a)
        self.b = Variable(b)
        self.c = Variable(c)
        self.x = Variable(x)
        self.y = Variable(y)
        self.output = F.relu(self.a*self.x + self.b*self.y + self.c)

    def math(self, a, b, c, x, y):
        n = a * x + b * y + c
        return max(n, 0.0)  # relu

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
        self.output.forward()
        return self.output.value

    def backward(self, grad=1.0):
        self.output.backward(grad)

    def zero_grad(self):
        self.output.zero_grad(backprop=True)

    def grad_descent(self, lr=0.01):
        self.a.set_value(-lr * self.a.grad, True)
        self.b.set_value(-lr * self.b.grad, True)
        self.c.set_value(-lr * self.c.grad, True)
        self.x.set_value(-lr * self.x.grad, True)
        self.y.set_value(-lr * self.y.grad, True)