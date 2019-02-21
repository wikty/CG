import math
import random


class Node(object):
    """Computational Graph Component: Node."""

    def __init__(self, value=None, grad=0.0, gate=None):
        """
        Params:
        `value` the value of node.
        `grad` the grad of node.
        `gate` the gate that produces node.

        You can access value and gradient via `value` and `grad` property.
        """
        self._value = value
        self._grad = grad
        self._gate = gate

    def __str__(self):
        return 'Node(%s/%s)' % (str(self._value), str(self._grad))

    @property
    def grad(self):
        return self._grad

    @property
    def value(self):
        return self._value

    def set_value(self, value, accumulated=False):
        """Change node value and not be tracked in graph."""
        if accumulated:
            self._value += value
        else:
            self._value = value
        return self

    def zero_grad(self, backprop=False):
        """Clear gradient."""
        self._grad = 0.0
        if backprop and self._gate:
            for node in self._gate.input():
                node.zero_grad(backprop)
        return self

    def forward(self):
        """Calculate and return node's value."""
        if self._gate is not None:
            # update predecessor nodes
            for node in self._gate.input():
                node.forward()
            # update current node
            self._value = self._gate.forward()
        return self

    def backward(self, grad=1.0):
        """Calculate node's gradient."""
        # update current node
        self._grad += grad  # accumulate gradients
        if self._gate is not None:
            # update predecessor nodes
            for n, g in zip(self._gate.input(), self._gate.backward()):
                n.backward(g)
        return self

    def dump_graph(self, tab_num=0):
        tab = '    '
        if self._gate is None:
            print('%s%s <- No Gate' % (tab*tab_num, self))
        else:
            print('%s%s <- %s(%s)' % (tab*tab_num, self, self._gate, 
                ', '.join([str(n) for n in self._gate.input()])
            ))
            for node in self._gate.input():
                node.dump_graph(tab_num+1)
        return self


class ConstNode(Node):
    """
    value is a constant.
    grad is always zero.
    gate is always none.
    """

    def __init__(self, value):
        self._value = value
        self._grad = 0.0
        self._gate = None

    def set_value(self, value, accumulate=False):
        return self

    def zero_grad(self, backprop=False):
        self._grad = 0.0
        return self

    def forward(self):
        return self

    def backward(self, grad=1.0):
        return self


class Gate(object):
    """Computational Graph Component: Gate."""

    def __init__(self, node1, node2=None):
        # input nodes
        self.in_node1 = node1
        self.in_node2 = node2
        self.in_len = 1 if node2 is None else 2
        # generate output node
        self.out_node = Node(None, 0.0, self)

    def __str__(self):
        return self.__class__.__name__

    def input(self):
        """Return a list of input nodes."""
        if self.in_len == 2:
            return [self.in_node1, self.in_node2]
        else:
            return [self.in_node1]

    def output(self):
        return self.out_node

    def forward(self):
        """Return the value of output node."""
        return 0.0

    def backward(self):
        """Return the gradients of input nodes."""
        return [None] * self.in_len


class AddGate(Gate):

    def forward(self):
        return self.in_node1.value + self.in_node2.value

    def backward(self):
        return [self.out_node.grad * 1.0, self.out_node.grad * 1.0]


class SubtractGate(Gate):

    def forward(self):
        return self.in_node1.value - self.in_node2.value

    def backward(self):
        return [self.out_node.grad * 1.0, self.out_node.grad * (-1.0)]


class MultiplyGate(Gate):

    def forward(self):
        return self.in_node1.value * self.in_node2.value

    def backward(self):
        return [
            self.out_node.grad * self.in_node2.value,
            self.out_node.grad * self.in_node1.value
        ]


class DivideGate(Gate):

    def forward(self):
        return self.in_node1.value / self.in_node2.value

    def backward(self):
        return [
            self.out_node.grad / self.in_node2.value,
            -(self.out_node.grad * self.in_node1.value) / (
                self.in_node2.value ** 2)
        ]


class PowGate(Gate):

    def __init__(self, node1, power=2):
        self.in_node1 = node1
        self.in_node2 = ConstNode(power)
        self.in_len = 2
        self.out_node = Node(None, 0.0, self)

    def forward(self):
        return math.pow(self.in_node1.value, self.in_node2.value)

    def backward(self):
        return [
            self.out_node.grad * (self.in_node2.value *
                math.pow(self.in_node1.value, self.in_node2.value-1)),
            0.0
        ]


class MaxGate(Gate):

    def forward(self):
        return max(self.in_node1.value, self.in_node2.value)

    def backward(self):
        if self.in_node1.value > self.in_node2.value:
            return [self.out_node.grad * 1.0, 0.0]
        else:
            return [0.0, self.out_node.grad * 1.0]

class MinGate(Gate):

    def forward(self):
        return min(self.in_node1.value, self.in_node2.value)

    def backward(self):
        if self.in_node1.value < self.in_node2.value:
            return [self.out_node.grad * 1.0, 0.0]
        else:
            return [0.0, self.out_node.grad * 1.0]


class ExpGate(Gate):

    def forward(self):
        return math.exp(self.in_node1.value)

    def backward(self):
        return [
            self.out_node.grad * math.exp(self.in_node1.value)
        ]


class SigmoidGate(Gate):

    def sigmoid(self, z):
        return (1 / (1 + math.exp(-z)))

    def forward(self):
        return self.sigmoid(self.in_node1.value)

    def backward(self):
        a = self.sigmoid(self.in_node1.value)
        return [self.out_node.grad * (a * (1 - a))]


class ReLUGate(Gate):

    def forward(self):
        return max(self.in_node1.value, 0.0)

    def backward(self):
        f = 1.0 if self.in_node1.value > 0.0 else 0.0
        return [self.out_node.grad * f]


class NegativeGate(Gate):

    def forward(self):
        return -(self.in_node1.value)

    def backward(self):
        return [self.out_node.grad * (-1.0)]


class PositiveGate(Gate):

    def forward(self):
        return +(self.in_node1.value)

    def backward(self):
        return [self.out_node.grad * 1.0]


class AbsoluteGate(Gate):

    def forward(self):
        return abs(self.in_node1.value)

    def backward(self):
        if self.in_node1.value > 0.0:
            return [self.out_node.grad * 1.0]
        else:
            return [self.out_node.grad * (-1.0)]


class Variable(object):
    """A wrapper for Node and Gates."""

    def __init__(self, value=None, node=None):
        self.node = Node(value, 0.0, None) if node is None else node

    @property
    def value(self):
        """Return variable value after forward running."""
        return self.node.value

    @property
    def grad(self):
        """Return variable grad after backward running."""
        return self.node.grad

    @classmethod
    def const(cls, value=0.0):
        return cls(node=ConstNode(value))

    def forward(self):
        """forward and return variable self.
        Run forward before access its value."""
        self.node.forward()
        return self

    def backward(self, grad=1.0):
        """Backward gradients and return variable self.
        Run forward and backward before access its grad.
        Maybe you should zero gradient first if you don't
        want to acculumate gradient.
        """
        self.node.backward(grad)
        return self

    def zero_grad(self, backprop=True):
        """Clear gradients and return variable self."""
        self.node.zero_grad(backprop)
        return self

    def set_value(self, value, accumulated=False):
        self.node.set_value(value, accumulated)
        return self

    def dump_graph(self, tab_num=0):
        self.node.dump_graph(tab_num)
        return self

    def _tovar(self, other):
        """Convert node or const into variable."""
        if isinstance(other, Variable):
            return other
        elif isinstance(other, Node):
            return Variable(node=other)
        else:
            # const
            return Variable(node=ConstNode(other))

    def __str__(self):
        return str(self.node.value)

    def __add__(self, other):
        """self+other, other is a variable or const."""
        other = self._tovar(other)
        return Variable(node=AddGate(self.node, other.node).output())

    def __radd__(self, other):
        """other+self, other is a const."""
        other = self._tovar(other)
        return Variable(node=AddGate(other.node, self.node).output())

    def __iadd__(self, other):
        """self+=other, other is a variable or const."""
        other = self._tovar(other)
        self.node = AddGate(self.node, other.node).output()
        return self

    def __sub__(self, other):
        """self-other, other is a variable or const."""
        other = self._tovar(other)
        return Variable(node=SubtractGate(self.node, other.node).output())

    def __rsub__(self, other):
        """other-self, other is a const."""
        other = self._tovar(other)
        return Variable(node=SubtractGate(other.node, self.node).output())

    def __isub__(self, other):
        """self-=other, other is a variable or const."""
        other = self._tovar(other)
        self.node = SubtractGate(self.node, other.node).output()
        return self

    def __mul__(self, other):
        """self*other, other is a variable or const."""
        other = self._tovar(other)
        return Variable(node=MultiplyGate(self.node, other.node).output())

    def __rmul__(self, other):
        """other*self, other is a const."""
        other = self._tovar(other)
        return Variable(node=MultiplyGate(other.node, self.node).output())

    def __imul__(self, other):
        """self*=other, other is a variable or const."""
        other = self._tovar(other)
        self.node = MultiplyGate(self.node, other.node).output()
        return self

    def __truediv__(self, other):
        """self/other, other is a variable or const."""
        other = self._tovar(other)
        return Variable(node=DivideGate(self.node, other.node).output())

    def __rtruediv__(self, other):
        """other/self, other is a const."""
        other = self._tovar(other)
        return Variable(node=DivideGate(other.node, self.node).output())

    def __itruediv__(self, other):
        """other/=self, other is a variable or const."""
        other = self._tovar(other)
        self.node = DivideGate(self.node, other.node).output()
        return self

    def __neg__(self):
        return Variable(node=NegativeGate(self.node).output())

    def __pos__(self):
        return Variable(node=PositiveGate(self.node).output())

    def __abs__(self):
        return Variable(node=AbsoluteGate(self.node).output())


class F(object):

    @staticmethod
    def square(v):
        return Variable(node=PowGate(v.node, 2).output())

    @staticmethod
    def max(v1, v2):
        return Variable(node=MaxGate(v1.node, v2.node).output())

    @staticmethod
    def min(v1, v2):
        return Variable(node=MinGate(v1.node, v2.node).output())

    @staticmethod
    def pow(v, power):
        assert isinstance(power, (int, float))
        return Variable(node=PowGate(v.node, power).output())

    @staticmethod
    def exp(v):
        return Variable(node=ExpGate(v.node).output())

    @staticmethod
    def sigmoid(v):
        return Variable(node=SigmoidGate(v.node).output())

    @staticmethod
    def relu(v):
        return Variable(node=ReLUGate(v.node).output())