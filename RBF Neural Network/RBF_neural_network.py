import os
import warnings
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs; 1 = filter out INFO logs; 2 = filter out WARNING logs; 3 = filter out ERROR logs
# Suppress specific TensorFlow warnings (if necessary)
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
# import TensorFlow
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from time import time
from keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
        nn.Linear(input_size, hidden_size1),
        nn.BatchNorm1d(hidden_size1),
        nn.ReLU(),
        nn.Linear(hidden_size1, hidden_size2),
        nn.BatchNorm1d(hidden_size2),
        nn.ReLU(),
        nn.Linear(hidden_size2, num_classes),
        )
        # self.linear = nn.Sequential(
        # nn.Linear(input_size, hidden_size1),
        # nn.BatchNorm1d(hidden_size1),
        # nn.ReLU(),
        # nn.Linear(hidden_size1, num_classes),
        # )

    def forward(self, x):
        return self.linear(x)

# Convert numpy arrays to PyTorch tensors
def numpy_to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

# Implement random center or k-means center calculator
def centers_calculator(n_centers, x_train):
    center_indexes = np.random.choice(x_train.shape[0], n_centers, replace=False)
    centers = x_train[center_indexes]
    return centers
    
# Implement gaussin rbf function for both gaussian and quadratic function
def rbf_function(x_train, class_center, parameter, type_of_rbf):
    if type_of_rbf == 'gaussian':
        return np.exp(-cdist(x_train, class_center, 'sqeuclidean') / (parameter**2))
    elif type_of_rbf == 'multiquadratic':
        return (np.sqrt(cdist(x_train, class_center, 'sqeuclidean') + parameter**2))

# Separate the classes from the PCA-transformed training dataset
def separate_classes(pca_train_data, y_train, num_classes=10):
    class_subsets = [pca_train_data[y_train == i] for i in range(num_classes)]
    return class_subsets

# Apply kmeans to everyone of the 10 classes of the PCA-transformed dataset
def apply_kmeans_to_classes(class_subsets, n_clusters, max_iters=300):
    kmeans_results = []
    for class_data in class_subsets:
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iters, random_state=0, n_init='auto')
        kmeans.fit(class_data)
        kmeans_results.append((kmeans.cluster_centers_, kmeans.labels_))
    return kmeans_results

def print_and_log(text, log_file):
    print(text)
    log_file.write(text + '\n')

# Load and preprocess training and test data for better and more efficient usage
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.reshape((-1, 3072)) / 255.0
x_test = x_test.reshape((-1, 3072)) / 255.0
y_train = y_train.ravel()
y_test = y_test.ravel()
x_train, y_train = shuffle(x_train, y_train, random_state=0)

# One-hot encode the labels
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded = to_categorical(y_test, 10)

# Use PCA to reduce the dimension of the data while keeping at least 90 percent of the information 
# In our case PCA transforms the dataset from 3072 to just 99 
pca = PCA(0.9).fit(x_train)
pca_train_data = pca.transform(x_train)
pca_test_data = pca.transform(x_test)

#Implement kmean clustering
class_subsets = separate_classes(pca_train_data, y_train)


#==========================================================================================================================================#
# Here is the part that trains using logistic regression function
# n_clusters_options = [5, 10, 20, 30, 40, 50, 100, 150, 200, 300]
n_clusters_options = [100, 150, 200, 300]
# parameter_options = [0.01, 0.1, 0.5, 1, 2, 3, 5, 8, 10, 12, 15, 20]
parameter_options = [1, 2, 3, 5, 8, 10, 12, 15, 20]
# type_of_rbf_options = ['gaussian', 'multiquadratic']
type_of_rbf = 'gaussian'

with open("data_log.txt", "w") as log_file:
    # Iterate over each combination of n_clusters and parameter
    for n_clusters in n_clusters_options:
        # Apply k-means clustering with current n_clusters
        kmeans_results = apply_kmeans_to_classes(class_subsets, n_clusters=n_clusters)
        # Extracting and combining the centroids from kmeans_results
        all_rbf_centers = np.vstack([centers for centers, _ in kmeans_results])

        for parameter in parameter_options:
            # for type_of_rbf in type_of_rbf_options:
            start_time = time()
            # Applying the Gaussian RBF function
            rbf_transformed_data_train = rbf_function(pca_train_data, all_rbf_centers, parameter, type_of_rbf=type_of_rbf)
            rbf_transformed_data_test = rbf_function(pca_test_data, all_rbf_centers, parameter, type_of_rbf=type_of_rbf)

            # Scale the RBF-transformed data
            scaler = StandardScaler()
            rbf_transformed_data_train_scaled = scaler.fit_transform(rbf_transformed_data_train)
            rbf_transformed_data_test_scaled = scaler.transform(rbf_transformed_data_test)

            # Train logistic regression model with increased max_iter
            logistic_regression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=20000)
            logistic_regression.fit(rbf_transformed_data_train_scaled, y_train)

            # Calculate and print accuracies
            train_accuracy = logistic_regression.score(rbf_transformed_data_train_scaled, y_train)
            test_accuracy = logistic_regression.score(rbf_transformed_data_test_scaled, y_test)

            # Prepare the result string
            output = (
                f"Number of clusters: {n_clusters}, Parameter: {parameter}, Type of RBF: {type_of_rbf}\n"
                f"Time collapsed: {time()-start_time:.2f} seconds\n"
                f"Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
            )
            print_and_log(output, log_file)

#==========================================================================================================================================#
# # Here is the part that trains using MLP
# n_clusters_options = [50, 100, 150]
# parameter_options = [8, 10, 12]
# type_of_rbf = 'gaussian'

# epochs = 100
# batch_size = 100

# hidden_size1 = 256
# hidden_size2 = 64
# num_classes = 10

# with open("data_log.txt", "w") as log_file:
#     # Iterate over each combination of n_clusters and parameter
#     for n_clusters in n_clusters_options:
#         # Apply k-means clustering with current n_clusters
#         kmeans_results = apply_kmeans_to_classes(class_subsets, n_clusters=n_clusters)
#         # Extracting and combining the centroids from kmeans_results
#         all_rbf_centers = np.vstack([centers for centers, _ in kmeans_results])

#         for parameter in parameter_options:
#             start_time = time()
#             # Applying the Gaussian RBF function
#             rbf_transformed_data_train = rbf_function(pca_train_data, all_rbf_centers, parameter, type_of_rbf=type_of_rbf)
#             rbf_transformed_data_test = rbf_function(pca_test_data, all_rbf_centers, parameter, type_of_rbf=type_of_rbf)

#             # Scale the RBF-transformed data
#             scaler = StandardScaler()
#             rbf_transformed_data_train_scaled = scaler.fit_transform(rbf_transformed_data_train)
#             rbf_transformed_data_test_scaled = scaler.transform(rbf_transformed_data_test)

#             # Convert data to PyTorch tensors
#             X_train_tensor = numpy_to_tensor(rbf_transformed_data_train_scaled)
#             Y_train_tensor = numpy_to_tensor(y_train_encoded)
#             X_test_tensor = numpy_to_tensor(rbf_transformed_data_test_scaled)
#             Y_test_tensor = numpy_to_tensor(y_test_encoded)

#             # Create DataLoader for batch processing
#             train_data = TensorDataset(X_train_tensor, Y_train_tensor)
#             test_data = TensorDataset(X_test_tensor, Y_test_tensor)
#             train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
#             test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

#             # Model, loss, and optimizer
#             model = MLP(X_train_tensor.shape[1], hidden_size1, hidden_size2, num_classes)
#             criterion = nn.CrossEntropyLoss()
#             optimizer = optim.Adam(model.parameters(), lr=0.001)

#             # Initialize lists to store metrics for plotting
#             epoch_numbers = []
#             test_accuracies = []
#             training_losses = []

#             # Training and evaluation loop
#             for epoch in range(epochs):
#                 # Training
#                 model.train()
#                 train_loss = 0.0
#                 train_correct = 0
#                 train_total = 0
#                 for i, (inputs, labels) in enumerate(train_loader):
#                     # Forward pass
#                     outputs = model(inputs)
#                     loss = criterion(outputs, torch.max(labels, 1)[1])
#                     train_loss += loss.item() * inputs.size(0)

#                     # Backward and optimize
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#                     _, predicted = torch.max(outputs.data, 1)
#                     train_total += labels.size(0)
#                     train_correct += (predicted == torch.max(labels, 1)[1]).sum().item()

#                 train_loss /= train_total
#                 train_accuracy = 100 * train_correct / train_total

#                 # Evaluation
#                 model.eval()
#                 test_correct = 0
#                 test_total = 0
#                 with torch.no_grad():
#                     for inputs, labels in test_loader:
#                         outputs = model(inputs)
#                         predicted = torch.max(outputs.data, 1)[1]
#                         test_total += labels.size(0)
#                         test_correct += (predicted == torch.max(labels, 1)[1]).sum().item()

#                 test_accuracy = 100 * test_correct / test_total

#                 # Store metrics
#                 epoch_numbers.append(epoch + 1)
#                 test_accuracies.append(test_accuracy)
#                 training_losses.append(train_loss)

#                 # Log training and test results
#                 print_and_log(f'Number of clusters per class: {n_clusters}, Parameter: {parameter}, Type of RBF: {type_of_rbf}\n'
#                             f'Epoch [{epoch+1}/{epochs}], '
#                             f'Training Loss: {train_loss:.4f}, '
#                             f'Training Accuracy: {train_accuracy:.2f}%, '
#                             f'Test Accuracy: {test_accuracy:.2f}%', log_file)
                
#             # Plotting Test Accuracy
#             plt.figure(figsize=(10, 6))
#             plt.plot(epoch_numbers, test_accuracies, label='Test Accuracy')
#             plt.xlabel('Epochs')
#             plt.ylabel('Test Accuracy (%)')
#             plt.title(f'Test Accuracy over Epochs\nn_clusters={n_clusters}, parameter={parameter}')
#             plt.legend()
#             plt.savefig(f'test_accuracy_nclusters{n_clusters}_param{parameter}.png')
#             plt.show()

#             # Plotting Training Loss
#             plt.figure(figsize=(10, 6))
#             plt.plot(epoch_numbers, training_losses, label='Training Loss')
#             plt.xlabel('Epochs')
#             plt.ylabel('Loss')
#             plt.title(f'Training Loss over Epochs\nn_clusters={n_clusters}, parameter={parameter}')
#             plt.legend()
#             plt.savefig(f'training_loss_nclusters{n_clusters}_param{parameter}.png')
#             plt.show()