from LinearModel import *
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = None, None, None, None
    # TODO define datasets properly
    print('Welcome to BreastCancerClassifier application! Please choose your desired model category and enter a number:'
          '\n1-LinearModel'
          '\n2-EnsembleModels and DecisionTrees'
          '\n3-SVM')
    cat = int(input())
    if cat == 1:
        clf = LinearClassifiers(x_train, y_train, x_test, y_test)
        print('choose among available linear models below:')
        print('1.Ridge Classifier')
        print('2. Logistic Regression')
        num = int(input())
        if num == 1:
            print('input alpha:')
            alpha = int(input())
            test_acc = clf.ridge_classifier(alpha)
        elif num == 2:
            print('input penalty (l1 or l2 ?):')
            penalty = input()
            test_acc = clf.logistic_regression(penalty)


