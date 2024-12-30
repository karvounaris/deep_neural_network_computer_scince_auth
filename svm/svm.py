import tensorflow as tf
from sklearn.svm import SVC
from time import time
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

def print_and_log(text, log_file):
    print(text)
    log_file.write(text + '\n')

with open("svm_log.txt", "w") as log_file:

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.reshape((-1, 3072)) / 255.0
    x_test = x_test.reshape((-1, 3072)) / 255.0

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    print_and_log('', log_file)



    for C in [0.1, 1, 10, 50, 100, 1000]:
        start_time = time()
        linearSVM = SVC(kernel='linear', C=C)
        linearSVM.fit(x_train[:5000], y_train[:5000])
        output = "Linear Kernel, C = " + str(C) + "\n"
        output += "Number of support vectors = " + str(linearSVM.n_support_) + "\n"
        output += "Elapsed time: " + str(time() - start_time) + "\n"
        output += "Train accuracy = " + str(linearSVM.score(x_train[:5000], y_train[:5000])) + "\n"
        output += "Test accuracy = " + str(linearSVM.score(x_test, y_test)) + "\n"
        print_and_log(output, log_file)

    for degree in [2, 3]:
        for C in [0.1, 1, 10, 50, 100, 1000]:
            for gamma in ['scale', 'auto', 0.1, 0.5, 1, 10, 20]:
                start_time = time()
                polySVM = SVC(kernel='poly', C=C, degree=degree, gamma=gamma)
                polySVM.fit(x_train[:5000], y_train[:5000])
                output = "Polynomial Kernel, Degree = " + str(degree) + ", C = " + str(C) + ", Gamma = " + str(gamma) + "\n"
                output += "Number of support vectors = " + str(polySVM.n_support_) + "\n"
                output += "Elapsed time: " + str(time() - start_time) + "\n"
                output += "Train accuracy = " + str(polySVM.score(x_train[:5000], y_train[:5000])) + "\n"
                output += "Test accuracy = " + str(polySVM.score(x_test, y_test)) + "\n"
                print_and_log(output, log_file)

    for C in [0.1, 1, 10, 50, 100, 1000]:
        for gamma in ['scale', 'auto', 0.1, 0.5, 1, 10, 20]:
            start_time = time()
            sigmoidSVM = SVC(kernel='sigmoid', C=C, gamma=gamma)
            sigmoidSVM.fit(x_train[:5000], y_train[:5000])
            output = "Sigmoid Kernel, C = " + str(C) + ", Gamma = " + str(gamma) + "\n"
            output += "Number of support vectors = " + str(sigmoidSVM.n_support_) + "\n"
            output += "Elapsed time: " + str(time() - start_time) + "\n"
            output += "Train accuracy = " + str(sigmoidSVM.score(x_train[:5000], y_train[:5000])) + "\n"
            output += "Test accuracy = " + str(sigmoidSVM.score(x_test, y_test)) + "\n"
            print_and_log(output, log_file)

    for C in [0.1, 1, 10, 50, 100, 1000]:
        for gamma in ['scale', 'auto', 0.1, 0.5, 1, 10, 20]:
            start_time = time()
            rbfSVM = SVC(kernel='rbf', C=C, gamma=gamma)
            rbfSVM.fit(x_train[:5000], y_train[:5000])
            output = "RBF Kernel, C = " + str(C) + ", Gamma = " + str(gamma) + "\n"
            output += "Number of support vectors = " + str(rbfSVM.n_support_) + "\n"
            output += "Elapsed time: " + str(time() - start_time) + "\n"
            output += "Train accuracy = " + str(rbfSVM.score(x_train[:5000], y_train[:5000])) + "\n"
            output += "Test accuracy = " + str(rbfSVM.score(x_test, y_test)) + "\n"
            print_and_log(output, log_file)






    for C in [0.1, 1, 10, 50, 100, 1000]:
        start_time = time()
        linearSVM = SVC(kernel='linear', C=C)
        linearSVM.fit(x_train[:10000], y_train[:10000])
        output = "Linear Kernel, C = " + str(C) + "\n"
        output += "Number of support vectors = " + str(linearSVM.n_support_) + "\n"
        output += "Elapsed time: " + str(time() - start_time) + "\n"
        output += "Train accuracy = " + str(linearSVM.score(x_train[:10000], y_train[:10000])) + "\n"
        output += "Test accuracy = " + str(linearSVM.score(x_test, y_test)) + "\n"
        print_and_log(output, log_file)

    for degree in [2, 3]:
        for C in [0.1, 1, 10, 50, 100, 1000]:
            for gamma in ['scale', 'auto', 0.1, 0.5, 1, 10, 20]:
                start_time = time()
                polySVM = SVC(kernel='poly', C=C, degree=degree, gamma=gamma)
                polySVM.fit(x_train[:10000], y_train[:10000])
                output = "Polynomial Kernel, Degree = " + str(degree) + ", C = " + str(C) + ", Gamma = " + str(gamma) + "\n"
                output += "Number of support vectors = " + str(polySVM.n_support_) + "\n"
                output += "Elapsed time: " + str(time() - start_time) + "\n"
                output += "Train accuracy = " + str(polySVM.score(x_train[:10000], y_train[:10000])) + "\n"
                output += "Test accuracy = " + str(polySVM.score(x_test, y_test)) + "\n"
                print_and_log(output, log_file)

    for C in [0.1, 1, 10, 50, 100, 1000]:
        for gamma in ['scale', 'auto', 0.1, 0.5, 1, 10, 20]:
            start_time = time()
            sigmoidSVM = SVC(kernel='sigmoid', C=C, gamma=gamma)
            sigmoidSVM.fit(x_train[:10000], y_train[:10000])
            output = "Sigmoid Kernel, C = " + str(C) + ", Gamma = " + str(gamma) + "\n"
            output += "Number of support vectors = " + str(sigmoidSVM.n_support_) + "\n"
            output += "Elapsed time: " + str(time() - start_time) + "\n"
            output += "Train accuracy = " + str(sigmoidSVM.score(x_train[:10000], y_train[:10000])) + "\n"
            output += "Test accuracy = " + str(sigmoidSVM.score(x_test, y_test)) + "\n"
            print_and_log(output, log_file)

    for C in [0.1, 1, 10, 50, 100, 1000]:
        for gamma in ['scale', 'auto', 0.1, 0.5, 1, 10, 20]:
            start_time = time()
            rbfSVM = SVC(kernel='rbf', C=C, gamma=gamma)
            rbfSVM.fit(x_train[:10000], y_train[:10000])
            output = "RBF Kernel, C = " + str(C) + ", Gamma = " + str(gamma) + "\n"
            output += "Number of support vectors = " + str(rbfSVM.n_support_) + "\n"
            output += "Elapsed time: " + str(time() - start_time) + "\n"
            output += "Train accuracy = " + str(rbfSVM.score(x_train[:10000], y_train[:10000])) + "\n"
            output += "Test accuracy = " + str(rbfSVM.score(x_test, y_test)) + "\n"
            print_and_log(output, log_file)






    pca = PCA(0.9).fit(x_train)
    pca_train_data = pca.transform(x_train)
    pca_test_data = pca.transform(x_test)

    for C in [0.1, 1, 10, 50, 100, 1000]:
        start_time = time()
        linearSVM = SVC(kernel='linear', C=C)
        linearSVM.fit(pca_train_data[:10000], y_train[:10000])
        output = "Linear Kernel, C = " + str(C) + "\n"
        output += "Number of support vectors = " + str(linearSVM.n_support_) + "\n"
        output += "Elapsed time: " + str(time() - start_time) + "\n"
        output += "Train accuracy = " + str(linearSVM.score(pca_train_data[:5000], y_train[:5000])) + "\n"
        output += "Test accuracy = " + str(linearSVM.score(pca_test_data, y_test)) + "\n"
        print_and_log(output, log_file)

    for degree in [2, 3]:
        for C in [0.1, 1, 10, 50, 100, 1000]:
            for gamma in ['scale', 'auto', 0.1, 0.5, 1, 10, 20]:
                start_time = time()
                polySVM = SVC(kernel='poly', C=C, degree=degree, gamma=gamma)
                polySVM.fit(pca_train_data[:10000], y_train[:10000])
                output = "Polynomial Kernel, Degree = " + str(degree) + ", C = " + str(C) + ", Gamma = " + str(gamma) + "\n"
                output += "Number of support vectors = " + str(polySVM.n_support_) + "\n"
                output += "Elapsed time: " + str(time() - start_time) + "\n"
                output += "Train accuracy = " + str(polySVM.score(pca_train_data[:10000], y_train[:10000])) + "\n"
                output += "Test accuracy = " + str(polySVM.score(pca_test_data, y_test)) + "\n"
                print_and_log(output, log_file)

    for C in [0.1, 1, 10, 50, 100, 1000]:
        for gamma in ['scale', 'auto', 0.1, 0.5, 1, 10, 20]:
            start_time = time()
            sigmoidSVM = SVC(kernel='sigmoid', C=C, gamma=gamma)
            sigmoidSVM.fit(pca_train_data[:10000], y_train[:10000])
            output = "Sigmoid Kernel, C = " + str(C) + ", Gamma = " + str(gamma) + "\n"
            output += "Number of support vectors = " + str(sigmoidSVM.n_support_) + "\n"
            output += "Elapsed time: " + str(time() - start_time) + "\n"
            output += "Train accuracy = " + str(sigmoidSVM.score(pca_train_data[:10000], y_train[:10000])) + "\n"
            output += "Test accuracy = " + str(sigmoidSVM.score(pca_test_data, y_test)) + "\n"
            print_and_log(output, log_file)

    for C in [0.1, 1, 10, 50, 100, 1000]:
        for gamma in ['scale', 'auto', 0.1, 0.5, 1, 10, 20]:
            start_time = time()
            rbfSVM = SVC(kernel='rbf', C=C, gamma=gamma)
            rbfSVM.fit(pca_train_data[:10000], y_train[:10000])
            output = "RBF Kernel, C = " + str(C) + ", Gamma = " + str(gamma) + "\n"
            output += "Number of support vectors = " + str(rbfSVM.n_support_) + "\n"
            output += "Elapsed time: " + str(time() - start_time) + "\n"
            output += "Train accuracy = " + str(rbfSVM.score(pca_train_data[:10000], y_train[:10000])) + "\n"
            output += "Test accuracy = " + str(rbfSVM.score(pca_test_data, y_test)) + "\n"
            print_and_log(output, log_file)






    C = 0.1
    start_time = time()
    linearSVM = SVC(kernel='linear', C=C)
    linearSVM.fit(pca_train_data[:50000], y_train[:50000])
    output = "Linear Kernel, C = " + str(C) + "\n"
    output += "Number of support vectors = " + str(linearSVM.n_support_) + "\n"
    output += "Elapsed time: " + str(time() - start_time) + "\n"
    output += "Train accuracy = " + str(linearSVM.score(pca_train_data[:50000], y_train[:50000])) + "\n"
    output += "Test accuracy = " + str(linearSVM.score(pca_test_data, y_test)) + "\n"
    print_and_log(output, log_file)

    C = 10
    gamma = 'scale'
    degree = 2
    coef0 = 0.5
    start_time = time()
    polySVM = SVC(kernel='poly', C=C, degree=degree, gamma=gamma, coef0 = coef0)
    polySVM.fit(pca_train_data[:50000], y_train[:50000])
    output = "Polynomial Kernel, Degree = " + str(degree) + ", C = " + str(C) + ", Gamma = " + str(gamma) + ", coef0 = " + str(coef0) + "\n"
    output += "Number of support vectors = " + str(polySVM.n_support_) + "\n"
    output += "Elapsed time: " + str(time() - start_time) + "\n"
    output += "Train accuracy = " + str(polySVM.score(pca_train_data[:50000], y_train[:50000])) + "\n"
    output += "Test accuracy = " + str(polySVM.score(pca_test_data, y_test)) + "\n"
    print_and_log(output, log_file)

    C = 10
    gamma = 0.5
    degree = 2
    coef0 = 0.5
    start_time = time()
    polySVM = SVC(kernel='poly', C=C, degree=degree, gamma=gamma, coef0 = coef0, probability = True)
    polySVM.fit(pca_train_data[:50000], y_train[:50000])
    output = "Polynomial Kernel, Degree = " + str(degree) + ", C = " + str(C) + ", Gamma = " + str(gamma) + ", coef0 = " + str(coef0) + ", probability = True" + "\n"
    output += "Number of support vectors = " + str(polySVM.n_support_) + "\n"
    output += "Elapsed time: " + str(time() - start_time) + "\n"
    output += "Train accuracy = " + str(polySVM.score(pca_train_data[:50000], y_train[:50000])) + "\n"
    output += "Test accuracy = " + str(polySVM.score(pca_test_data, y_test)) + "\n"
    print_and_log(output, log_file)

    gamma = 'scale'
    for C in [1, 10, 100, 1000]:
        start_time = time()
        rbfSVM = SVC(kernel='rbf', C=C, gamma=gamma)
        rbfSVM.fit(pca_train_data[:50000], y_train[:50000])
        output = "RBF Kernel, C = " + str(C) + ", Gamma = " + str(gamma) + "\n"
        output += "Number of support vectors = " + str(rbfSVM.n_support_) + "\n"
        output += "Elapsed time: " + str(time() - start_time) + "\n"
        output += "Train accuracy = " + str(rbfSVM.score(pca_train_data[:50000], y_train[:50000])) + "\n"
        output += "Test accuracy = " + str(rbfSVM.score(pca_test_data, y_test)) + "\n"
        print_and_log(output, log_file)
    
    gamma = 'scale'
    for C in [1, 10, 100, 1000]:
        start_time = time()
        rbfSVM = SVC(kernel='rbf', C=C, gamma=gamma, decision_function_shape = "ovo")
        rbfSVM.fit(pca_train_data[:50000], y_train[:50000])
        output = "RBF Kernel, C = " + str(C) + ", Gamma = " + str(gamma) + ", decision_function_shape = \"ovo\"" + "\n"
        output += "Number of support vectors = " + str(rbfSVM.n_support_) + "\n"
        output += "Elapsed time: " + str(time() - start_time) + "\n"
        output += "Train accuracy = " + str(rbfSVM.score(pca_train_data[:50000], y_train[:50000])) + "\n"
        output += "Test accuracy = " + str(rbfSVM.score(pca_test_data, y_test)) + "\n"
        print_and_log(output, log_file)