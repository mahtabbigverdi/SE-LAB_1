from Ensembles import *
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = None, None, None,None
    print('Welcome to BreastCancerClassifier application! Please choose your desired model category and enter a number:'
          '\n1-LinearModel'
          '\n2-EnsembleModels and DecisionTrees'
          '\n3-SVM')
    cat = int(input())

    if cat == 2:
        clf = EnsembleAndDTClassifiers(x_train, y_train, x_test, y_test)
        print('choose among available ensemble models below:')
        print('1. Random Forest')
        print('2. Decisoin Tree')
        num = int(input())
        if num == 1:
            print('input numer of estimators:')
            n_estimators = int(input())
            tets_acc = clf.random_forest(n_estimators = n_estimators)
        if num == 2:
            tets_acc = clf.decision_tree()


