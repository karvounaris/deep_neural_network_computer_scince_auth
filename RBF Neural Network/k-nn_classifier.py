import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#load batches
batch1 = unpickle('cifar-10-batches-py/data_batch_1')
batch2 = unpickle('cifar-10-batches-py/data_batch_2')
batch3 = unpickle('cifar-10-batches-py/data_batch_3')
batch4 = unpickle('cifar-10-batches-py/data_batch_4')
batch5 = unpickle('cifar-10-batches-py/data_batch_5')
batchT = unpickle('cifar-10-batches-py/test_batch')

x_train = np.concatenate([batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data']])
y_train = np.concatenate([batch1[b'labels'], batch2[b'labels'], batch3[b'labels'], batch4[b'labels'], batch5[b'labels']])

# Extracting data and labels for test batch
x_test = batchT[b'data']
y_test = batchT[b'labels']

# Normalize the pixel values to a range between 0 and 1, for better performance for k-nn
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Flatten the images
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  
knn.fit(x_train_flat, y_train)

# Predict labels for the test data
y_pred = knn.predict(x_test_flat)

# Calculate accuracy, recall, and F1 score
accuracy = knn.score(x_test_flat, y_test)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Generate a classification report
class_report = classification_report(y_test, y_pred)

print("k-Nearest Neighbours Classifier Results:")
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1)
print(class_report)
