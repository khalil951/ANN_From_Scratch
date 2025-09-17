import numpy as np
import matplotlib.pyplot as plt
from gemini_api import generate_synthetic_data

np.random.seed(0)

def create_data(points, classes):
    X = np.zeros((points*classes, 2)) # data matrix (each row = single example)
    y = np.zeros(points*classes, dtype='uint8') # class labels
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0,1,points) # radius
        t = np.linspace(class_number*4,(class_number+1)*4,points) + np.random.randn(points)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = class_number
    return X, y


print
X,y = create_data(100,3)
plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral)
plt.show()


inputs = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]

weights1 = [0.2,0.8,-0.5,1.0]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

# output = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
#           inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
#           inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3]

# layer_outputs = [
#     float(f'{sum(input_val * weight for input_val, weight in zip(inputs, weights_row)) + bias_val:.2f}')
#     for weights_row, bias_val in zip(weights, bias)
# ]


weights = [weights1, weights2, weights3]
biases = [bias1, bias2, bias3]

weights_2 = [[0.1,-0.14,0.5],
             [-0.5,0.12,-0.33],
             [-0.44,0.73,-0.13]]
biases_2 = [-1,2,-0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights_2).T) + biases_2
print(f'Layer of outputs is {layer2_outputs} with shape {layer2_outputs.shape}')