import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons, learning_rate, batch_size):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)   
        self.bias = np.zeros((1, n_neurons))
        self.delta = np.zeros((batch_size, n_neurons))
        self.learning_rate = learning_rate
 
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias
    
    # def back_propagation_output(self, inputs, y_pred, y_train, batch_size, learning_rate):
    #     self.learning_rate = learning_rate
    #     self.delta = (y_pred - y_train) * y_pred * (1 - y_pred)
    #     #  mx10        mx10       mx10     mx10           mx10
    #     weights = self.learning_rate * ((self.delta).T).dot(inputs) / batch_size
    #     #  10x3072                                 10xm           mx3072
    #     self.weights -= weights.T
    #     #  3072x10
    #     self.bias -= self.learning_rate * np.mean((self.delta).T, axis = 1)
    #     #  1x10                              1x10                                    

    # def back_propagation_hidden(self, ReLU_derivatives, previous_weights, previous_delta, inputs, batch_size, learning_rate):
    #     self.learning_rate = learning_rate
    #     self.delta = previous_delta.dot(previous_weights.T) * ReLU_derivatives  # mx3072
    #     #  mx3072        mx10                  10x3072           mx3072 
    #     weights = self.learning_rate * (inputs.T).dot(self.delta) / batch_size
    #     #  3072x3072                     3072xm         mx3072
    #     self.weights -= weights
    #     #  3072x3072
    #     self.bias -= self.learning_rate * np.mean((self.delta), axis = 0)
    #     #  1x3072                                mx3072



    def back_propagation_output(self, y_pred, y_train, learning_rate):
        self.learning_rate = learning_rate
        self.delta = (y_pred - y_train) * y_pred * (1 - y_pred)
        #  mx10        mx10       mx10     mx10           mx10
    
    def update_param_output(self, inputs, batch_size):
        self.weights = self.weights - (self.learning_rate * ((self.delta).T).dot(inputs) / batch_size).T
        #  3072x10
        self.bias -= self.learning_rate * np.mean((self.delta).T, axis = 1)
        #  1x10                              1x10                                    

    def back_propagation_hidden(self, ReLU_derivatives, previous_weights, previous_delta, learning_rate):
        self.learning_rate = learning_rate
        self.delta = previous_delta.dot(previous_weights.T) * ReLU_derivatives  # mx3072
        #  mx3072        mx10                  10x3072           mx3072 

    def update_param_hidden(self, inputs, batch_size):
        self.weights = self.weights - self.learning_rate * (inputs.T).dot(self.delta) / batch_size
        #  3072x10
        self.bias -= self.learning_rate * np.mean((self.delta).T, axis = 1)
        #  1x10


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    
    def derivative_ReLu(self,inputs):
        return (inputs > 0).astype(int)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        samples_losses = self.forward(output, y)
        data_loss = np.mean(samples_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_train):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_train.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_train]
        elif len(y_train.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_train, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot_encode(labels, num_classes):
    one_hot_encoded = np.zeros((len(labels), num_classes))
    one_hot_encoded[np.arange(len(labels)), labels] = 1
    return one_hot_encoded

def normalize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < 1e-8] = 1e-8

    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std

    return X_train_normalized, X_test_normalized

def normalize_data_single(X_train):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < 1e-8] = 1e-8
    X_train_normalized = (X_train - mean) / std
    return X_train_normalized

def get_predictions(y_pred):
    return np.argmax(y_pred, axis=1)

def get_accuracy(predictions, y_test):
    print(np.sum(predictions == y_test))
    print(len(y_test))
    return np.sum(predictions == y_test) / len(y_test)

learning_rate = 1e-3
# learning_rate_1 = 1e-3
# learning_rate_2 = 1e-4
batch_size = 400
number_of_epochs = 15
accuracy_array = []
loss_array = []
epoch_array = []

# In case of cifar-10 is already in the directory
# batch1 = unpickle('cifar-10-batches-py/data_batch_1')
# batch2 = unpickle('cifar-10-batches-py/data_batch_2')
# batch3 = unpickle('cifar-10-batches-py/data_batch_3')
# batch4 = unpickle('cifar-10-batches-py/data_batch_4')
# batch5 = unpickle('cifar-10-batches-py/data_batch_5')
# batchT = unpickle('cifar-10-batches-py/test_batch')

# x_train = np.concatenate([batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data']])
# y_train = np.concatenate([batch1[b'labels'], batch2[b'labels'], batch3[b'labels'], batch4[b'labels'], batch5[b'labels']])
# y_train_en = one_hot_encode(y_train, 10)

# x_test = batchT[b'data']
# y_test = batchT[b'labels']

# x_train, x_test = normalize_data(x_train,x_test)

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Flatten the labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Normalize the pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images for dense layers
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# One-hot encode the labels
y_train_en = one_hot_encode(y_train, 10)

input_layer = Dense_Layer(3072, 3072, learning_rate, batch_size)
activation_relu_input = Activation_ReLU()
inside_layer = Dense_Layer(3072, 200, learning_rate, batch_size)
activation_relu_inside = Activation_ReLU()
output_layer = Dense_Layer(200, 10, learning_rate, batch_size)
activation_softmax = Activation_Softmax()
loss_function = Loss_CategoricalCrossEntropy()

for epoch in range(number_of_epochs):
    train_samples, features = x_train.shape
    for j in range(int(train_samples / batch_size)):
        X_train = x_train[batch_size * j : (batch_size * (j+1))]
        Y_train = y_train[batch_size * j : (batch_size * (j+1))]
        Y_train_en = y_train_en[batch_size * j : (batch_size * (j+1))]
        
        input_layer.forward(X_train)
        activation_relu_input.forward(input_layer.output)
        activation_relu_input.output = normalize_data_single(activation_relu_input.output)

        inside_layer.forward(activation_relu_input.output)
        activation_relu_inside.forward(inside_layer.output)
        activation_relu_inside.output = normalize_data_single(activation_relu_inside.output)

        output_layer.forward(activation_relu_inside.output)
        activation_softmax.forward(output_layer.output)

        # output_layer.back_propagation_output(activation_relu_inside.output, activation_softmax.output, Y_train_en, batch_size, learning_rate)
        # relu_derivatives = activation_relu_inside.derivative_ReLu(inside_layer.output)
        # inside_layer.back_propagation_hidden(relu_derivatives, output_layer.weights, output_layer.delta, activation_relu_input.output, batch_size, learning_rate)
        # relu_derivatives = activation_relu_input.derivative_ReLu(input_layer.output)
        # input_layer.back_propagation_hidden(relu_derivatives, inside_layer.weights, inside_layer.delta, X_train, batch_size, learning_rate)

        output_layer.back_propagation_output(activation_softmax.output, Y_train_en, learning_rate)
        relu_derivatives = activation_relu_inside.derivative_ReLu(inside_layer.output)
        inside_layer.back_propagation_hidden(relu_derivatives, output_layer.weights, output_layer.delta, learning_rate)
        relu_derivatives = activation_relu_input.derivative_ReLu(input_layer.output)
        input_layer.back_propagation_hidden(relu_derivatives, inside_layer.weights, inside_layer.delta, learning_rate)
        output_layer.update_param_output(activation_relu_inside.output, batch_size)
        inside_layer.update_param_hidden(activation_relu_input.output, batch_size)
        input_layer.update_param_hidden(X_train, batch_size)

    if(epoch % 5 == 0 or epoch == number_of_epochs - 1):
        input_layer.forward(x_train)
        activation_relu_input.forward(input_layer.output)
        activation_relu_input.output = normalize_data_single(activation_relu_input.output)

        inside_layer.forward(activation_relu_input.output)
        activation_relu_inside.forward(inside_layer.output)
        activation_relu_inside.output = normalize_data_single(activation_relu_inside.output)

        output_layer.forward(activation_relu_inside.output)
        activation_softmax.forward(output_layer.output)
        
        loss = loss_function.calculate(activation_softmax.output, y_train_en)
        accuracy_train = get_accuracy(get_predictions(activation_softmax.output), y_train)
        loss_array.append(loss)
        accuracy_array.append(accuracy_train)
        epoch_array.append(epoch)
        print("Epoch: " + str(epoch))
        print("Loss: " + str(loss))
        print("Train accuracy: " + str(accuracy_train))

        input_layer.forward(x_test)
        activation_relu_input.forward(input_layer.output)
        activation_relu_input.output = normalize_data_single(activation_relu_input.output)

        inside_layer.forward(activation_relu_input.output)
        activation_relu_inside.forward(inside_layer.output)
        activation_relu_inside.output = normalize_data_single(activation_relu_inside.output)

        output_layer.forward(activation_relu_inside.output)
        activation_softmax.forward(output_layer.output)

        accuracy_test = get_accuracy(get_predictions(activation_softmax.output), y_test)
        print("Test accuracy: " + str(accuracy_test))

    # if epoch == 10:
    #     learning_rate = learning_rate_1
    # elif epoch == 100:
    #     learning_rate = learning_rate_2

input_layer.forward(x_test)
activation_relu_input.forward(input_layer.output)
activation_relu_input.output = normalize_data_single(activation_relu_input.output)

inside_layer.forward(activation_relu_input.output)
activation_relu_inside.forward(inside_layer.output)
activation_relu_inside.output = normalize_data_single(activation_relu_inside.output)

output_layer.forward(activation_relu_inside.output)
activation_softmax.forward(output_layer.output)

accuracy_test = get_accuracy(get_predictions(activation_softmax.output), y_test)
print("Test accuracy: " + str(accuracy_test))


plt.figure()
plt.plot(epoch_array, accuracy_array)
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.figure()
plt.plot(epoch_array, loss_array)
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()

