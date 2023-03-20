from utils import utils

import numpy
import pandas as pd
import numpy as np
import keras_metrics as km
from time import time
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.utils import to_categorical
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.decomposition import PCA
# from plsda import PLSDA
# from simca import SIMCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from FTIR_KNN import TestKNN
from PLS import airPLS
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
# scores = [cohen_kappa_score,f1_score, accuracy_score, recall_score,precision_score]
from plsda import PLSDA
from sklearn.ensemble import VotingClassifier

polymerName, waveLength, intensity, polymerID=utils.parseDataForSecondDataset('new_SecondDataset.csv')
PCAintensity=PCA(n_components=200).fit_transform(intensity)
PCAintensity=np.array(PCAintensity)
print(PCAintensity.shape)
print(intensity)
print(polymerID)
SVM_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
SVM_clf_PCA = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=1)
x_train_PCA, x_test_PCA, y_train_PCA, y_test_PCA = train_test_split(PCAintensity, polymerID, test_size=0.3, random_state=1)
# SVM_clf.fit(x_train,y_train)
# SVM_clf_PCA.fit(x_train_PCA,y_train_PCA)
# print('~~~~~',SVM_clf.score(x_test,y_test))
# print('~~~~~',SVM_clf_PCA.score(x_test_PCA,y_test_PCA))
# MLPmodel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 64, 64), random_state=1)
KNNModel = KNeighborsClassifier(n_neighbors=2)
# KNNModel.fit(x_train,y_train)
# print('~~~~~',KNNModel.score(x_test,y_test))
KNNModelPCA = KNeighborsClassifier(n_neighbors=2)
# KNNModelPCA.fit(x_train_PCA,y_train_PCA)
# print('~~~~~',KNNModelPCA.score(x_test_PCA,y_test_PCA))
LDA_clf = LinearDiscriminantAnalysis(n_components=4, priors=None, shrinkage=None,
                                     solver='svd', store_covariance=True, tol=0.0001)
LDA_clf_PCA = LinearDiscriminantAnalysis(n_components=4, priors=None, shrinkage=None,
                                     solver='svd', store_covariance=True, tol=0.0001)
# LDA_clf.fit(x_train,y_train)
# LDA_clf_PCA.fit(x_train_PCA,y_train_PCA)
# print('~~~~~',LDA_clf.score(x_test,y_test))
# print('~~~~~',LDA_clf_PCA.score(x_test_PCA,y_test_PCA))
RandomF = RandomForestClassifier(n_estimators=74, oob_score=True,
                                 random_state=10, max_depth=11,
                                 min_samples_split=2, min_samples_leaf=2, max_features=9)
RandomF_PCA = RandomForestClassifier(n_estimators=74, oob_score=True,
                                 random_state=10, max_depth=11,
                                 min_samples_split=2, min_samples_leaf=2, max_features=9)
# RandomF.fit(x_train,y_train)
# RandomF_PCA.fit(x_train_PCA,y_train_PCA)
# print('~~~~~',RandomF.score(x_test,y_test))
# print('~~~~~',RandomF_PCA.score(x_test_PCA,y_test_PCA))
SVM_clf = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo',probability=True)
SVM_clfPCA = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo',probability=True)
RandomF = RandomForestClassifier(n_estimators=74, oob_score=True,
                                 random_state=10, max_depth=11,
                                 min_samples_split=2, min_samples_leaf=2, max_features=9)
LDA_clf_PCA = LinearDiscriminantAnalysis(n_components=4, priors=None, shrinkage=None,
                                     solver='svd', store_covariance=True, tol=0.0001)
sp = PLSDA(n_components=30, style='hard')

cmTotal = np.zeros((5, 5))

PCAModel = load('model/pca200.joblib')
#models = [SVM_clf, MLPmodel, KNNModel, RandomF, LDA_clf, sp]

totalReport = []
PN =[]
for item in polymerName:
    if item not in PN:
        PN.append(item)
maxnumber=[]
for p in range(1):
    m = 0
    t_report = []
    scoreTotal = np.zeros(5)
    scoreTotal2 =np.zeros(5)
    scoreTotal3 = np.zeros(5)
    for i in range(10):
        totalProbability = []
        totalProbabilityForTest = []

        preName = []
        preID = []
        polymerName, waveLength, intensity, polymerID = utils.parseDataForSecondDataset('new_SecondDataset.csv')
        PCAintensity = PCA(n_components=200).fit_transform(intensity)
        x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=i)
        x_train_PCA, x_test_PCA, y_train_PCA, y_test_PCA = train_test_split(PCAintensity, polymerID, test_size=0.3, random_state=i)


        test = set(y_train.tolist())
        print(len(test))
        if(len(test)<5):
            test=[]
            continue
        # model = tS.spectrumSVM(x_train, y_train, 0.3, 'linear', 'ovo')
        # KNNModelPCA.fit(x_train_PCA,y_train_PCA)
        # sp.fit(x_train_PCA,y_train_PCA)
        # LDA_clf_PCA.fit(x_train_PCA,y_train_PCA)
        Voting1 = VotingClassifier(estimators=[
            ('KNN', KNNModelPCA),  ('LDA', LDA_clf_PCA)],
            voting='soft')
        SVM_clf.fit(x_train,y_train)
        SVM_clfPCA.fit(x_train_PCA,y_train_PCA)
        KNNModel.fit(x_train,y_train)
        KNNModelPCA.fit(x_train_PCA,y_train_PCA)
        Voting1.fit(x_train_PCA,y_train_PCA)
        RandomF.fit(x_train,y_train)
        KNNpre = KNNModel.predict(x_test)
        SVMpre = SVM_clf.predict(x_test)

        whetherBreak = 0

        for t in range(len(x_train)):
            totalPre = []
            pre1 = SVM_clf.predict_proba([x_train[t]])
            pre2 = SVM_clfPCA.predict_proba([x_train_PCA[t]])
            pre3 = KNNModel.predict_proba([x_train[t]])
            pre4 = KNNModelPCA.predict_proba([x_train_PCA[t]])
            pre5 = Voting1.predict_proba([x_train_PCA[t]])
            pre6 = RandomF.predict_proba([x_train[t]])
            totalPre.append(pre1)
            totalPre.append(pre2)
            totalPre.append(pre3)
            totalPre.append(pre4)
            totalPre.append(pre5)
            totalPre.append(pre6)
            totalPre=np.array(totalPre)
            totalPre=totalPre.reshape(30,)
            totalProbability.append(totalPre)
        for u in range(len(x_test)):
            totalPrefortest = []
            pre1 = SVM_clf.predict_proba([x_test[u]])
            pre2 = SVM_clfPCA.predict_proba([x_test_PCA[u]])
            pre3 = KNNModel.predict_proba([x_test[u]])
            pre4 = KNNModelPCA.predict_proba([x_test_PCA[u]])
            pre5 = Voting1.predict_proba([x_test_PCA[u]])
            pre6 = RandomF.predict_proba([x_test[u]])
            totalPrefortest.append(pre1)
            totalPrefortest.append(pre2)
            totalPrefortest.append(pre3)
            totalPrefortest.append(pre4)
            totalPrefortest.append(pre5)
            totalPrefortest.append(pre6)

            totalPrefortest=np.array(totalPrefortest)

            totalPrefortest=totalPrefortest.reshape(30,)
            totalProbabilityForTest.append(totalPrefortest)


        polymerID = to_categorical(polymerID)
        x_train2, x_test2, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3,
                                                              random_state=i)

        x_train =numpy.array(totalProbability)

        x_test = numpy.array(totalProbabilityForTest)



        def deep_model(feature_dim, label_dim):
            from keras.models import Sequential
            from keras.layers import Dense
            model = Sequential()
            print("create model. feature_dim ={}, label_dim ={}".format(feature_dim, label_dim))
            model.add(Dense(64, activation='relu', input_dim=feature_dim))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(label_dim, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model


        def train_deep(X_train, y_train, X_test, y_test):
            feature_dim = X_train.shape[1]
            label_dim = 5

            model = deep_model(feature_dim, label_dim)
            model.summary()
            model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))
            return model


        model = train_deep(x_train, y_train, x_test, y_test)
        results=model.predict(x_test)
        results=results.tolist()
        for item in results:
            preID.append(item.index(max(item)))
#         for j in range(len(y_test)):
#             y_test[j]=int(y_test[j])
#
        cm = confusion_matrix(y_test_PCA, preID)
        print(cm)
        cmtemp = np.zeros((5, 5))
        if len(cm) < 5:
            print('stop')
            continue
            # for i in range(len(cm)):
            #     cmtemp[i]=np.append(cm[i],[0],axis=0)
            #     # cmtemp[i]+=cm[i]
            # cmtemp[11,11]=2
            # print('tempCM；',cmtemp)
            # cmTotal+=cmtemp
            # continue
        cmTotal = cmTotal + cm

        scores = utils.printScore(y_test_PCA, preID)
        scores2 = utils.printScore(y_test_PCA,KNNpre)
        scores3= utils.printScore(y_test_PCA,SVMpre)
        print(scores)
        print('KNN',scores2)
        print('SVM', scores3)
        m += 1
        print(m)
        scoreTotal += scores
        scoreTotal2 +=scores2
        scoreTotal3+=scores3
        rep = classification_report(y_test_PCA, preID, target_names=PN, output_dict=True)
        utils.mkdir('workflow5_report_NN')
        workflow4_report = pd.DataFrame(rep)
        print(workflow4_report)
        workflow4_report.to_csv('workflow5_report_NN/workflow' + str(i) + '.csv')
#
#         # t = classification_report(int(y_test), preID, target_names=PN, output_dict=True)
#         #
#         # SVM_report = pd.DataFrame(t)
#         # print(SVM_report)
#
#
    print(scoreTotal / m)
    print(scoreTotal2/m)
    print(scoreTotal3 / m)
    maxnumber.append(sum(scoreTotal/m) )
#
    cmTotal = cmTotal / m
    print(cmTotal / m)
    utils.plot_confusion_matrix(cmTotal,PN,'Jung Dataset')
#
indexVector=maxnumber.index(max(maxnumber))
print(max(maxnumber),indexVector)