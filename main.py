import os
import sys
# sys.path.append(os.path.abspath(r"//Users/shyhhao/Documents/AMLSassignment19_-20_LSH_SN16067637/AMLS_19-20_LSH_SN16067637/A1"))
# sys.path.append(os.path.abspath(r"//Users/shyhhao/Documents/AMLSassignment19_-20_LSH_SN16067637/AMLS_19-20_LSH_SN16067637/A2"))
# sys.path.append(os.path.abspath(r"//Users/shyhhao/Documents/AMLSassignment19_-20_LSH_SN16067637/AMLS_19-20_LSH_SN16067637/B1"))
# sys.path.append(os.path.abspath(r"//Users/shyhhao/Documents/AMLSassignment19_-20_LSH_SN16067637/AMLS_19-20_LSH_SN16067637/B2"))

for extension in ['A1', 'A2', 'B1', 'B2']:
    sys.path.append(os.getcwd() + '/' + extension)

from Gender_A1 import *
from Smiling_A2 import *
from Faceshape_B1 import *
from EyeColor_B2 import *
# ======================================================================================================================
# Task A1: Run A1 accuracy test
# from datetime import datetime
# start = datetime.now()    
# # tr_X_A1, tr_Y_A1, te_X_A1, te_Y_A1 = get_data_A1()
# # model_A1 = A1_SVM(tr_X_A1, list(zip(*tr_Y_A1))[0], te_X_A1, list(zip(*te_Y_A1))[0])
# # acc_A1_train, acc_A1_test, pred_A1 = model_A1
# # print('TA1:{},{}'.format(acc_A1_train, acc_A1_test))
# print(datetime.now() - start)

# ======================================================================================================================
# Task A2: Run A2 accuracy test
# from datetime import datetime
# start = datetime.now()
# # tr_X_A2, tr_Y_A2, te_X_A2, te_Y_A2= get_data_A2()
# # model_A2 = A2_SVM(tr_X_A2, list(zip(*tr_Y_A2))[0], te_X_A2, list(zip(*te_Y_A2))[0])
# # acc_A2_train, acc_A2_test, pred_A2 = model_A2
# # print('TA2:{},{}'.format(acc_A2_train, acc_A2_test))
# print(datetime.now() - start)

# ======================================================================================================================
# Task B1: Run B1 accuracy test
# SVM Function
# from datetime import datetime
# start = datetime.now()    
# # tr_X_B1, tr_Y_B1, te_X_B1, te_Y_B1= get_data_B1()
# # model_B1 = B1_SVM(tr_X_B1, list(zip(*tr_Y_B1))[0], te_X_B1, list(zip(*te_Y_B1))[0])
# # acc_B1_train, acc_B1_test, pred_B1 = model_B1
# # print('TB1:{},{}'.format(acc_B1_train, acc_B1_test))
# print(datetime.now() - start)

# CNN
from datetime import datetime
acc_B1 = accuracy_score(true_B1, pred_B1)
rec_B1 = recall_score(true_B1, pred_B1, pos_label = 'positive', average ='macro') ## Weighted using macro
pre_B1 = precision_score(true_B1,pred_B1, pos_label = 'positive', average ='macro') ## Weighted using macro
f1_B1 = f1_score(true_B1,pred_B1, pos_label = 'positive', average ='macro') ## Weighted using macro
print(datetime.now() - start)
# ======================================================================================================================
# Task B2: Run B2 accuracy test
# SVM Function
# from datetime import datetime
# start = datetime.now()   
# # tr_X_B2, tr_Y_B2, te_X_B2, te_Y_B2= get_data_B2()
# # model_B2 = B2_SVM(tr_X_B2, list(zip(*tr_Y_B2))[0], te_X_B2, list(zip(*te_Y_B2))[0])
# # acc_B2_train, acc_B2_test, pred_B2 = model_B2
# # print('TB2:{},{}'.format(acc_B2_train, acc_B2_test))
# # print(datetime.now() - start)

# CNN
from datetime import datetime
start = datetime.now()
acc_B2 = accuracy_score(true_B2, pred_B2)
rec_B2 = recall_score(true_B2, pred_B2, pos_label = 'positive', average ='macro') ## Weighted using macro
pre_B2 = precision_score(true_B2,pred_B2, pos_label = 'positive', average ='macro') ## Weighted using macro
f1_B2 = f1_score(true_B2,pred_B2, pos_label = 'positive', average ='macro') ## Weighted using macro
print(datetime.now() - start)

# ======================================================================================================================
## Print SVM results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

## Print CNN Accuracy for B1 and B2
print("Accuracy_B1 :" + str(acc_B1))
print("Precision_B1 :" + str(pre_B1))
print("Recall_B1 :" + str(rec_B1))
print("F1 Score_B1 :" + str(f1_B1))

print("Accuracy_B2 :" + str(acc_B2))
print("Precision_B2 :" + str(pre_B2))
print("Recall_B2 :" + str(rec_B2))
print("F1 Score_B2 :" + str(f1_B2))
