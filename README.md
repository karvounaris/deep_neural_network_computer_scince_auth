# 🤖 Deep Neural Network Projects

## 📖 Overview
This repository includes three distinct projects developed as part of the Neural Networks course at Aristotle University of Thessaloniki. Each project explores advanced neural network techniques applied to the CIFAR-10 dataset:

1. **Support Vector Machines (SVM)**
2. **Radial Basis Function (RBF) Neural Network**
3. **Multilayer Perceptron (MLP) Neural Network**

The projects highlight different machine learning techniques, emphasizing their strengths and limitations for image classification tasks.

---

## 🎯 Goals

- **SVM Project**:
  - Evaluate the performance of Support Vector Machines using different kernels and parameter configurations.
  - Incorporate PCA for dimensionality reduction and improve training efficiency.
- **RBF Neural Network Project**:
  - Implement RBF networks with optimized clustering strategies for robust classification.
  - Compare logistic regression and MLP as output layers.
- **MLP Neural Network Project**:
  - Design and train an MLP from scratch for CIFAR-10 classification.
  - Explore the impact of hyperparameter tuning and architectural changes.

---

## ✨ Features

### 📊 Support Vector Machines (SVM)
- **Dataset**: CIFAR-10 images, preprocessed and scaled.
- **Kernel Types**: Linear, Polynomial, Sigmoid, and RBF.
- **Dimensionality Reduction**: PCA applied to reduce features from 3072 to 100 dimensions while retaining 90% of the data variance.
- **Results**:
  - **Best Accuracy**: Achieved using RBF kernel with specific parameter settings (e.g., \( C = 10, 	ext{gamma} = 	ext{scale} \)).
  - Significant improvement in training time with PCA.

### 🌫️ Radial Basis Function Neural Network
- **Clustering**:
  - K-means clustering applied to individual classes for better centroid initialization.
  - Adaptive neuron allocation to improve classification.
- **Output Layers**:
  - **Logistic Regression**: Used as a baseline for performance comparison.
  - **MLP**: Integrated as an output layer with varying architectures.
- **Results**:
  - Best performance observed with Gaussian RBF functions and a well-tuned MLP output layer.

### 🧠 Multilayer Perceptron (MLP) Neural Network
- **Implementation**: Built from scratch using Python, inspired by the mathematical analysis in Haykin’s Neural Networks.
- **Architectural Details**:
  - Initial design: One hidden layer with 128 neurons.
  - Final design: Two hidden layers with 200 neurons each.
- **Challenges Addressed**:
  - Debugging backpropagation errors.
  - Mitigating overfitting through hyperparameter tuning and architectural refinements.
- **Results**:
  - Achieved significant accuracy improvement after addressing logical errors and optimizing learning rates.

---

## 🏆 Results
- **SVM**: Achieved competitive accuracy for CIFAR-10 classification with efficient PCA preprocessing.
- **RBF Neural Network**: Demonstrated the effectiveness of tailored clustering strategies and output layer combinations.
- **MLP Neural Network**: Successfully built and trained a custom MLP, achieving meaningful insights into neural network operations.

---

## 🛠️ Techniques Utilized
- **Dimensionality Reduction**: PCA for feature space optimization.
- **Kernel Methods**: Comparative analysis of SVM kernels.
- **Neural Networks**: Comprehensive design and training of MLP and RBF networks.
- **Clustering**: Advanced K-means strategies for neuron initialization.
- **Hyperparameter Tuning**: Systematic exploration to optimize performance.

---

## 📂 Repository Contents
- **📄 Reports**:
  - [Support Vector Machines Report](./SVM_10193.pdf)
  - [RBF Neural Network Report](./RBF_Neural_Network.pdf)
  - [MLP Neural Network Report](./Neurolink_Network_1.pdf)
- **💻 Code**: Python scripts for implementing and testing all three projects.
- **📊 Results**: Detailed performance metrics and visualizations.

---

## 🤝 Contributors
- [Panagiotis Karvounaris](https://github.com/karvounaris)

---

Thank you for exploring these projects! 🌟 Feel free to raise issues or contribute to improve the repository. 🚀😊
