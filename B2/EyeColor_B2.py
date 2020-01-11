# Importing necessary libraries for SVM Function
import os
import sys
sys.path.append(os.path.abspath(r"//Users/shyhhao/Documents/AML_Assignment/AMLSassignment19_-20_SN16067637/B2"))
import B2_landmarks as b2
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

# Get features and labels data from B2_landmark.py and spilt them into train and test datasets
def get_data_B2():

    X_B2, y_B2 = b2.extract_features_labels()
    Y_B2 = np.array([y_B2, -(y_B2 - 1)]).T

    # Rescaling Data
    scaler = StandardScaler()
    temp_X_reshape = X_B2.reshape(len(X_B2), len(X_B2[0]) * len(X_B2[0][0]))
    temp_X_B2 = scaler.fit_transform(temp_X_reshape)

    tr_X_B2, te_X_B2, tr_Y_B2, te_Y_B2 = train_test_split(temp_X_B2, Y_B2, test_size=0.3, random_state=0)

#     print(tr_X_B2.shape)
#     print(te_X_B2.shape)
#     print(tr_Y_B2.shape)
#     print(te_Y_B2.shape)
    
    return tr_X_B2, tr_Y_B2, te_X_B2, te_Y_B2

# SVM function with 3 different kernels and various parameters
def B2_SVM(training_images, training_labels, test_images, test_labels):
#     classifier = svm.SVC(kernel='linear', C = 0.05)
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100]},
                        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C':[1, 10, 100]},
                        {'kernel': ['poly'], 'degree': [2, 3], 'C': [1, 10, 100]}
                        ]
    classifier = GridSearchCV(svm.SVC(), tuned_parameters, n_jobs = -1)
#     lab_enc = preprocessing.LabelEncoder()
    classifier.fit(training_images, training_labels)
                                
#  SVM Accuracy
    acc_B2_train = classifier.best_score_

#  Choosing the best accuracy using GridSearchCV
    pred_B2 = classifier.best_estimator_.predict(test_images)
    acc_B2_test = accuracy_score(test_labels, pred_B2)
    
    print(classifier.best_estimator_)
    print()
    print(classifier.best_params_)
    
    return acc_B2_train, acc_B2_test, pred_B2
    
# Plotting learning curve
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
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
# Learning Curve
title = "Learning Curves SVM"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = svm.SVC(kernel='rbf', gamma=1e-3, C=100)
X, y = tr_X_B2, list(zip(*tr_Y_B2))[0]
plot_learning_curve(estimator, title, X, y, (0, 0.6), cv=cv, n_jobs=-1)

plt.show()

# Confusion Matrix
test_label = list(zip(*te_Y_B2))[0]
cf = confusion_matrix(test_label, pred_B2)
# print(cf)
cmap = plt.cm.Blues

plt.matshow(cf, cmap = cmap)
plt.title('Confusion matrix')
plt.colorbar()
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show

# Moving on to CNN as SVM produces very low accuracy and lots of image pre-processing functions to do before running through SVM
# USING CNN
# Importing libraries for CNN
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import pandas as pd
from sklearn import svm
from keras.preprocessing.image import ImageDataGenerator
srcdir = '/Users/shyhhao/Documents/AMLSassignment19_-20_LSH_SN16067637/AMLS_19-20_LSH_SN16067637/dataset_AMLS_19-20/dataset_AMLS_19-20/cartoon_set/img'

# Generating dataframe from CSV
df = pd.read_csv("/Users/shyhhao/Documents/AMLSassignment19_-20_LSH_SN16067637/AMLS_19-20_LSH_SN16067637/dataset_AMLS_19-20/dataset_AMLS_19-20/cartoon_set/labels.csv")
df = pd.DataFrame(df).reset_index()
df.columns = ['Index', 'Total']
del df['Index']
df['eye_color'] = df['Total'].str.split('\t').str[1]
df['face_shape'] = df['Total'].str.split('\t').str[2]
df['img_name'] = df['Total'].str.split('\t').str[3]
del df['Total']
df
# Appending labels to respective colors
eyescolor = []
for i in range(len(df.eye_color)):
    if df.eye_color.loc[i] == '0':
        eyescolor.append('brown')
    elif df.eye_color.loc[i] == '1':
        eyescolor.append('blue')
    elif df.eye_color.loc[i] == '2':
        eyescolor.append('green')
    elif df.eye_color.loc[i] == '3':
        eyescolor.append('gray')
    elif df.eye_color.loc[i] == '4':
        eyescolor.append('black')
df['colors'] = eyescolor
df

# Splitting data
train, test = train_test_split(df, test_size = 0.2, random_state = 0)
print(len(train))
print(len(test))

# Generating training and validation data
xcol = 'img_name'
ycol = 'colors'
print('Receiving data..')
# Creating ariticial datas based on original dataset
data = ImageDataGenerator(rescale = 1./255.,
                          validation_split = 0.25,
                          horizontal_flip = True,
                          vertical_flip = True
                         )

print('Arranging training dataset..')
train_gen = data.flow_from_dataframe(dataframe = train,
                                     directory = srcdir,
                                     x_col = xcol,
                                     y_col = ycol,
                                     class_mode = 'categorical',
                                     target_size = (64,64),
                                     batch_size = 32,
                                     subset = 'training'
                                    )

print('Arranging validation dataset..')
val_gen = data.flow_from_dataframe(dataframe = train,
                                   directory = srcdir,
                                   x_col = xcol,
                                   y_col = ycol,
                                   class_mode = 'categorical',
                                   target_size = (64,64),
                                   batch_size = 32,
                                   subset = 'validation'
                                  )

# Importing libraries used for CNN
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
# Layer filter matrixes for CNN
model = Sequential()

model.add(Conv2D(24, (3,3), input_shape=train_gen.image_shape))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(24, (3,3)))
model.add(Activation("relu"))

model.add(Conv2D(48, (3,3)))
model.add(Activation("relu"))

model.add(Conv2D(96, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(5))
model.add(Activation("softmax"))

opt = optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print(model.summary())

# Plotting learning curve of accuracy based on CNN history
num_epochs = 25
epoch_nums = range(1,num_epochs+1)
training_acc = cnn_training.history["acc"]
validation_acc = cnn_training.history["val_acc"]
plt.figure(figsize=(10,8))
plt.plot(epoch_nums, training_acc)
plt.plot(epoch_nums, validation_acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['training', 'validation'], loc='lower right')
plt.show()

# Confusion matrix
print("Generating predictions from validation data..")

x_test = val_gen[0][0]
y_test = val_gen[0][1]

class_probabilities = model.predict(x_test)

predictions = np.argmax(class_probabilities, axis=1)

true_labels = np.argmax(y_test, axis=1)

classes = ["Brown", "Blue", "Green", "Gray" , "Black"]

cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8,10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes , rotation=85)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# TESTING NEW DATASET
newdir = '/Users/shyhhao/Documents/dataset_test_AMLS_19-20/cartoon_set_test/img'
df_new = pd.read_csv("/Users/shyhhao/Documents/dataset_test_AMLS_19-20/cartoon_set_test/labels.csv")
df_new = pd.DataFrame(df_new).reset_index()
df_new.columns = ['Index', 'Total']
del df_new['Index']
df_new['eye_color'] = df_new['Total'].str.split('\t').str[1]
df_new['face_shape'] = df_new['Total'].str.split('\t').str[2]
df_new['img_name'] = df_new['Total'].str.split('\t').str[3]
del df_new['Total']
del df_new['face_shape']
df_new

eyescolor = []
for i in range(len(df_new.eye_color)):
    if df_new.eye_color.loc[i] == '0':
        eyescolor.append('brown')
    elif df_new.eye_color.loc[i] == '1':
        eyescolor.append('blue')
    elif df_new.eye_color.loc[i] == '2':
        eyescolor.append('green')
    elif df_new.eye_color.loc[i] == '3':
        eyescolor.append('gray')
    elif df_new.eye_color.loc[i] == '4':
        eyescolor.append('black')
df_new['colors'] = eyescolor
df_new

xcol = 'img_name'
ycol = 'colors'
test_newdata = ImageDataGenerator(rescale=1./255)
val_gen_new = test_newdata.flow_from_dataframe(dataframe = df_new,
                                       directory = newdir,
                                       x_col = xcol,
                                       y_col = ycol,
                                       class_mode = 'categorical',
                                       target_size = (64,64),
                                       batch_size = 1,
                                       shuffle = False
                                      )

new_filenames = val_gen_new.filenames
nb_samples = len(new_filenames)

new_prob = model.predict_generator(val_gen_new,steps = nb_samples)
print(len(new_filenames), len(new_prob))

new_pred = np.argmax(new_prob, axis=1)
new_true = np.array(val_gen_new.classes)
print(len(new_pred))

newcm = confusion_matrix(new_true, new_pred)

print(newcm)

plt.imshow(newcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
# tick_marks = np.arange(len(classes))
plt.title('New Confusion Matrix')
plt.xlabel("New Predicted Class")
plt.ylabel("New True Class")
plt.show()

acc = accuracy_score(new_true, new_pred)
rec = recall_score(new_true, new_pred, pos_label = 'positive', average ='macro') ## Weighted using macro
pre = precision_score(new_true,new_pred, pos_label = 'positive', average ='macro') ## Weighted using macro
f1 = f1_score(new_true,new_pred, pos_label = 'positive', average ='macro') ## Weighted using macro
print("Accuracy :" + str(acc))
print("Precision :" + str(pre))
print("Recall :" + str(rec))
print("F1 Score :" + str(f1))

