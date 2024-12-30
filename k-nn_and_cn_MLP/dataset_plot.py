import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Function to load CIFAR-10 dataset from extracted files
def load_cifar10_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
        images = data[b'data']
        labels = data[b'labels']
        images = np.reshape(images, (len(images), 3, 32, 32)).transpose(0, 2, 3, 1)
        return images, labels

# Load CIFAR-10 data
cifar10_data_path = os.path.abspath('/home/karvounaris/Documents/university/neuralink network/dataset/cifar-10-batches-py')
x_train, y_train = load_cifar10_data(os.path.join(cifar10_data_path, 'data_batch_1'))

# Visualize some random images from the dataset
num_images_to_show = 7
random_indices = np.random.randint(0, len(x_train), num_images_to_show)

for i, idx in enumerate(random_indices):
    plt.subplot(1, num_images_to_show, i + 1)
    plt.imshow(x_train[idx])
    plt.title(f"Label: {y_train[idx]}")
    plt.axis('off')

plt.show()

# Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train.reshape(x_train.shape[0], -1))

# Plot classes in 2D
plt.figure(figsize=(10, 8))
for i in range(10):
    indices = np.where(np.array(y_train) == i)
    plt.scatter(x_train_pca[indices, 0], x_train_pca[indices, 1], label=f'Class {i}', alpha=0.5)

plt.title('CIFAR-10 Classes in 2D (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()