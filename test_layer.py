import unittest
import numpy as np
from layer import Activation_ReLU

class TestActivationReLU(unittest.TestCase):

    def setUp(self):
        """Set up a new Activation_ReLU instance for each test."""
        self.relu_activation = Activation_ReLU()

    def test_backward_pass_all_positive_inputs(self):
        """
        Tests the backward pass when all inputs to the ReLU function
        during the forward pass were positive.
        """
        # Simulate a forward pass with only positive inputs
        self.relu_activation.forward(np.array([[1, 5, 10]]))

        # Gradient from the subsequent layer
        dvalues = np.array([[0.1, -0.5, 1.2]])

        # Perform the backward pass
        result = self.relu_activation.backward(dvalues)

        # For positive inputs, the gradient should be passed back unchanged
        np.testing.assert_array_almost_equal(result, dvalues)

    def test_backward_pass_all_negative_or_zero_inputs(self):
        """
        Tests the backward pass when all inputs to the ReLU function
        during the forward pass were negative or zero.
        """
        # Simulate a forward pass with negative and zero inputs
        self.relu_activation.forward(np.array([[-1, 0, -10]]))

        # Gradient from the subsequent layer
        dvalues = np.array([[0.1, -0.5, 1.2]])

        # Perform the backward pass
        result = self.relu_activation.backward(dvalues)

        # For negative/zero inputs, the gradient should be zeroed out
        expected_dinputs = np.array([[0.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected_dinputs)

    def test_backward_pass_mixed_inputs(self):
        """Tests the backward pass with a mix of positive and negative inputs."""
        self.relu_activation.forward(np.array([[-2, 3, -5, 8]]))
        dvalues = np.array([[1.1, 2.2, 3.3, 4.4]])
        result = self.relu_activation.backward(dvalues)
        expected_dinputs = np.array([[0.0, 2.2, 0.0, 4.4]])
        np.testing.assert_array_almost_equal(result, expected_dinputs)

if __name__ == '__main__':
    unittest.main()
