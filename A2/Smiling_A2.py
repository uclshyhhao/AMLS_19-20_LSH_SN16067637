# Import all necessary libraries
import os
import sys
sys.path.append(os.path.abspath(r"//Users/shyhhao/Documents/AML_Assignment/AMLSassignment19_-20_SN16067637/A2"))
import A2_landmarks as a2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Get features and labels data from A2_landmark.py and spilt them into train and test datasets
def get_data_A2():

    X_A2, y_A2 = a2.extract_features_labels()
    Y_A2 = np.array([y_A2, -(y_A2 - 1)]).T
    
    # Rescaling Data
    scaler = StandardScaler()
    temp_X_reshape = X_A2.reshape(len(X_A2), len(X_A2[0]) * len(X_A2[0][0]))
    temp_X_A2 = scaler.fit_transform(temp_X_reshape)
    
    #     print(tr_X.shape)
    #     print(te_X.shape)
    #     print(tr_Y.shape)
    #     print(te_Y.shape)
    
    tr_X_A2, te_X_A2, tr_Y_A2, te_Y_A2 = train_test_split(temp_X_A2, Y_A2, test_size=0.3, random_state=0)
    
    return tr_X_A2, tr_Y_A2, te_X_A2, te_Y_A2

# SVM function with 3 different kernels and various parameters
def A2_SVM(training_images, training_labels, test_images, test_labels):
#     classifier = svm.SVC(kernel='linear', C = 0.05)
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100]},
                        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C':[1, 10, 100]},
                        {'kernel': ['poly'], 'degree': [2, 3], 'C': [1, 10, 100]}
                        ]
    classifier = GridSearchCV(svm.SVC(), tuned_parameters, n_jobs = -1)
    classifier.fit(training_images, training_labels)
                                
#  SVM Accuracy
    acc_A2_train = classifier.best_score_

#  Choosing the best accuracy using GridSearchCV
    pred_A2 = classifier.best_estimator_.predict(test_images)
    acc_A2_test = accuracy_score(test_labels, pred_A2)
    
    print(classifier.best_estimator_)
    print()
    print(classifier.best_params_)
    
    return acc_A2_train, acc_A2_test, pred_A2

# Plotting learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt    
# Learning Curve
title = "Learning Curves SVM"

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = svm.SVC(kernel='rbf', gamma=1e-4, C=100)
X, y = tr_X_A2, list(zip(*tr_Y_A2))[0]
plot_learning_curve(estimator, title, X, y, (0.85, 0.95), cv=cv, n_jobs=-1)

plt.show()

# Confusion Matrix
test_label = list(zip(*te_Y_A2))[0]
cf = confusion_matrix(test_label, pred_A2)
print(cf)

cmap = plt.cm.Blues

plt.matshow(cf, cmap = cmap)
plt.title('Confusion matrix')
plt.colorbar()
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show