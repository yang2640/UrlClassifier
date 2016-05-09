from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.learning_curve import learning_curve
from sklearn.multiclass import OneVsRestClassifier
import cPickle as pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
# import matplotlib.pyplot as plt


# load the feature, label
# split the dataset
# build the svm
# cross_validation, grid search parameters
# test and classification report


class Classifier:
    def __init__(self, data, label):
        self.savingpath = 'clf.pkl'
        # splits
        self.data_train, self.data_test, self.label_train, self.label_test = \
            train_test_split(data, label, test_size=0.2)


    # cross_validation
    def cross_validation(self):
        clf = OneVsRestClassifier(SVC(kernel='linear', C=10))
        scores = cross_val_score(clf,  # estimator
                                 self.data_train,  # training data
                                 self.label_train,  # training labels
                                 cv=5,  # split data randomly into 5 folders
                                 scoring='accuracy',  # which scoring metric?
                                 n_jobs=-1,  # -1 = use all cores = faster
                                 )

        print scores.mean(), scores.std()


    def grid_search(self):
        param_svm = [{'classifier__C': [1, 10, 100, 1000]}, ]
        grid_svm = GridSearchCV(
            clf,  # estimator
            param_grid=param_svm,  # parameters to tune via cross validation
            refit=True,  # fit using all data, on the best detected classifier
            n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
            scoring='accuracy',  # what score are we optimizing?
            cv=StratifiedKFold(self.label_train, n_folds=5),  # what type of cross validation to use
        )
        # print out the best parameters
        detector = grid_svm.fit(self.data_train, self.label_train)
        print detector.grid_scores_

    # train classifier and save
    def train(self):
        self.clf = OneVsRestClassifier(SVC(kernel='linear', C=10))
        self.clf.fit(self.data_train, self.label_train)

        # save the model to current dir
        with open(self.savingpath, 'wb') as f:
            pickle.dump(self.clf, f, pickle.HIGHEST_PROTOCOL)


    # final test
    def test(self, use_cache_model=False):
        if use_cache_model:
            with open('clf.pkl', 'rb') as f:
                self.clf = pickle.load(f)

        prediciton = self.clf.predict(self.data_test)

        # compare to label_test and generate the report and confusion matrix
        print 'precision:', precision_score(self.label_test, prediciton, average=None)

        # confusion matrix
        print 'confusion_matrix:'
        print confusion_matrix(self.label_test, prediciton)

if __name__ == '__main__':
    # test the classifier performance, load the feature vectors and labels
    with open('feat.pkl', 'rb') as f:
        feats = pickle.load(f)

    with open('label.pkl', 'rb') as f:
        label = pickle.load(f)

    classifer = Classifier(feats, label)
    classifer.train()
    classifer.test(True)
