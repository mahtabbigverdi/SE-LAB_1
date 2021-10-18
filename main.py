from SVM import *
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = None, None, None, None
    print('Welcome to BreastCancerClassifier application! Please choose your desired model category and enter a number:'
          '\n1-LinearModel'
          '\n2-EnsembleModels and DecisionTrees'
          '\n3-SVM')
    cat = int(input())

    if cat == 3:
        clf = Svm(x_train, y_train, x_test, y_test)
        print('input kernel: ')
        kernel = input()
        test_acc = clf.svm_classifier(kernel=kernel)
