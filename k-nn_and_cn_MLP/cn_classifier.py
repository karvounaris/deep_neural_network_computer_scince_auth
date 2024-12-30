from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
from sklearn.decomposition import PCA
import tensorflow as tf

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the pixel values to a range between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Flatten labels (convert from 2D to 1D array)
y_train = y_train.flatten()
y_test = y_test.flatten()

# Create and train the Nearest Centroid Classifier
centroid_classifier = NearestCentroid()
centroid_classifier.fit(x_train_flat, y_train)

# Predict labels for the test data
y_pred = centroid_classifier.predict(x_test_flat)

# Calculate accuracy, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Generate a classification report
class_report = classification_report(y_test, y_pred)

# Print the results
print("Nearest Centroid Classifier Results:")
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1)
print(class_report)
