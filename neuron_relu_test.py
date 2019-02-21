import unittest

from neuron_relu import Neuron


class TestNeuron(unittest.TestCase):

    def setUp(self):
        self.h = 0.001
        self.lr = 0.01
        self.args = [1, 2, 3, 4, 5]
        self.model = Neuron(*self.args)
        self.model.forward()

    def test_forward(self):
        self.assertAlmostEqual(self.model.forward(), 
                               self.model.math(*self.args))

    def test_backward(self):
        self.model.zero_grad()
        self.model.backward()
        for g1, g2 in zip(self.model.grad(), 
                          self.model.math_grad(*self.args, self.h)):
            self.assertAlmostEqual(g1, g2)

    def test_gradient_descent(self):
      before_descent = self.model.forward()
      self.model.grad_descent(self.lr)
      after_descent = self.model.forward()
      self.assertGreaterEqual(before_descent, after_descent)


if __name__ == '__main__':
    unittest.main()